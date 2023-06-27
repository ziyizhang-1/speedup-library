#  Copyright 2021 Intel Corporation
#  This software and the related documents are Intel copyrighted materials, and your use of them is governed by the
#  express license under which they were provided to you (License). Unless the License provides otherwise, you may not
#  use, modify, copy, publish, distribute, disclose or transmit this software or the related documents without Intel's
#  prior written permission.
#
#  This software and the related documents are provided as is, with no express or implied warranties, other than those
#  that are expressly stated in the License.
#
#
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List
from multiprocess import Pool, Manager

import cli.args
from cli.args import parse_args
from cli.args import ValidMultiFileSpec
from cli.timer import timer
from cli.tps import TPSViewGenerator
from cli.writers import excel_writer
from cli.writers.csv import CSVWriter
from cli.writers.views import ViewWriter
from cli import __version__
from cli.output_type import OutputInfo
from pedp import (
    EmonParser,
    EmonSystemInformationAdapter,
    JsonConstantParser,
    MetricDefinitionParserFactory,
    MetricComputer,
    Normalizer,
    RawEmonDataFrameColumns as redc,
    ViewType,
    ViewAggregationLevel,
    ViewCollection,
    ViewGenerator,
    DataAccumulator,
    Device
)


def main(dry_run=False):
    _print_banner()
    args = parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s',
                            datefmt='%Y-%m-%d %I:%M:%S %p', stream=sys.stderr)
    try:
        if not dry_run:
            run(args)
    except Exception as e:
        logging.exception(e)
        _error('Error running EDP', exception=e)


class SymbolTable:

    def __init__(self, system_info, core_type: str, latency_file_map: Dict):
        self.__symbols = EmonSystemInformationAdapter(system_info).get_symbol_table()
        self.__latency_symbols: Dict[str, float] = {}
        self._parse_latency_symbols(latency_file_map, core_type)
        self.__symbols.update(self.__latency_symbols)

    def _parse_latency_symbols(self, latency_file_map, core_type, latency_descriptor: str = 'MEAN') -> Dict[str, int]:
        if latency_file_map and core_type in latency_file_map.keys():
            json_constant_parser = JsonConstantParser(latency_file_map[core_type], latency_descriptor)
            self.__latency_symbols = json_constant_parser.parse()

    def symbols(self):
        return self.__symbols


def run(args):
    # Initialize EMON parser
    parser = EmonParser(Path(args.emon_dat_file_path),
                        emon_v_file=Path(args.emonv_file_path) if args.emonv_file_path else None,
                        ref_tsc_hz=args.frequency * 1000000 if args.frequency else 0)
    valid_device_names = list(parser.system_info.unique_core_types.copy())
    Device.set_valid_device_names(parser.system_info.unique_core_types)
    ValidMultiFileSpec.validate_multi_file_core_types(args.metric_file_path, parser.system_info.unique_core_types)
    output_info = OutputInfo(args.output_file_path)

    total_metrics_parsed = 0
    devices: List[Device] = []
    default_metric_computer = None
    requested_aggregation_levels = _get_requested_aggregation_levels(args)
    for core_type, metric_file in args.metric_file_path.items():
        # Initialize metric computer(s)
        symbol_table = SymbolTable(parser.system_info, core_type, args.retire_latency).symbols()
        metric_definitions = MetricDefinitionParserFactory.create(Path(metric_file)).parse()
        metric_computer = MetricComputer(metric_definitions, symbol_table)
        if default_metric_computer is None:
            default_metric_computer = metric_computer
        total_metrics_parsed += len(metric_definitions)
        devices.append(Device(core_type, requested_aggregation_levels, metric_computer))
    if args.uncore_view:
        devices = _append_uncore_devices(devices, valid_device_names, parser, default_metric_computer)

    # Initialize view generator
    view_collection = _initialize_views(args, parser, devices)
    view_generator = ViewGenerator(view_collection)
    data_accumulator = DataAccumulator(view_generator)

    csv_writer = CSVWriter(output_info.output_path)
    view_writer = ViewWriter(parser.system_info.has_modules, writers=csv_writer)

    print('System information:')
    print(f'     TSC Frequency: {parser.system_info.ref_tsc // 1000000} MHz')
    print('')
    print('Processing EMON data: ...', end='', flush=True)
    unique_events = set()
    num_blocks = len(parser.partition(chunk_size=1, **cli.args.get_sample_range_args(args)))
    if args.num_workers > 1:
        args.chunk_size = num_blocks // args.num_workers + (1 if num_blocks%args.num_workers!=0 else 0)
    
    # Split the EMON file into partitions that can be processed in parallel
    partitions = parser.partition(chunk_size=args.chunk_size,
                                  **cli.args.get_sample_range_args(args))

    def process_one_partition(idx, partition, offsets):
        # Read the entire partition into memory
        emon_event_reader = parser.event_reader(partition=partition, chunk_size=0)
        emon_df = next(emon_event_reader)
        sample_count = parser.last_sample_processed - parser.very_first_sample_processed + 1
        
        if detail_views_requested:
            detail_views = view_generator.generate_detail_views(emon_df)
            if idx > 0:
                while idx not in offsets.keys():
                    time.sleep(0.1)
                offset = offsets[idx]
                mode = 'a'
                print_header = False
            else:
                offset = 0
                mode = 'w'
                print_header = True
            
            view_writer.write(list(detail_views.values()),
                            first_sample=parser.first_sample_processed,
                            last_sample=parser.last_sample_processed,
                            mode=mode,
                            print_header=print_header)
            offsets[idx + 1] = offset + sample_count
            
        return emon_df, sample_count

    with timer() as number_of_seconds:
        # Process each partitions independently and serialize all computation results.
        # Iterations are independent and can therefore run in parallel.
        if args.num_workers > 1:
            detail_views_requested = not args.no_detail_views
            offsets = Manager().dict()
            with Pool() as pool:
                results = pool.starmap(process_one_partition, [(idx, partition, offsets) for idx, partition in enumerate(partitions)])
            sample_count = 0
            for emon_df, count in results:
                unique_events = unique_events.union(set(emon_df['name']))
                data_accumulator.update_statistics(df=emon_df)
                # Compute summary data for the partition and update it in the accumulator
                event_aggregates = view_generator.compute_aggregates(emon_df)
                data_accumulator.update_aggregates(event_aggregates)
                sample_count += count
        else:        
            for partition in partitions:
                # Read the entire partition into memory
                emon_event_reader = parser.event_reader(partition=partition, chunk_size=0)
                emon_df = next(emon_event_reader)
                unique_events = unique_events.union(set(emon_df['name']))
                print('.', end='', flush=True)
                detail_views_requested = not args.no_detail_views
                # Generate partial detail views for the partition and write to storage
                if detail_views_requested:
                    detail_views = view_generator.generate_detail_views(emon_df)
                    data_accumulator.update_statistics(detail_views)
                    view_writer.write(list(detail_views.values()),
                                    first_sample=parser.first_sample_processed,
                                    last_sample=parser.last_sample_processed)
                else:
                    data_accumulator.update_statistics(df=emon_df)
                # Compute summary data for the partition and update it in the accumulator
                event_aggregates = view_generator.compute_aggregates(emon_df)
                data_accumulator.update_aggregates(event_aggregates)
            sample_count = parser.last_sample_processed - parser.very_first_sample_processed + 1
        print()
        print(f'Generated all details tables and computed summary aggregations in {number_of_seconds}')

    summary_views_list = []
    with timer() as number_of_seconds:
        # Generate the summary views
        summary_views = data_accumulator.generate_summary_views()
        view_writer.write(list(summary_views.values()),
                          first_sample=parser.very_first_sample_processed,
                          last_sample=parser.last_sample_processed)
        summary_views_list.append(summary_views)
        print(f'Generated all summary tables in {number_of_seconds}')

    if args.transactions_per_second:
        with timer() as number_of_seconds:
            for views in summary_views_list:
                tps_generator = TPSViewGenerator(args.transactions_per_second)
                tps_summary_views = tps_generator.generate_summaries(views)
                view_writer.write(list(tps_summary_views.values()),
                                  first_sample=parser.very_first_sample_processed,
                                  last_sample=parser.last_sample_processed)
                tps_views = [value.attributes for _, value in tps_summary_views.items()]
                print(f'Generated all Transactions Per Second tables in {number_of_seconds}')

    print(f'{sample_count} samples processed')
    print(f'{len(unique_events)} events parsed and {total_metrics_parsed} metrics derived.')

    if output_info.output_excel:
        print('Creating Excel output file...')
        if args.transactions_per_second:
            view_collection.append_views(tps_views)
        excel_writer.write_csv_data_to_excel(args, view_collection)


def _get_requested_aggregation_levels(args):
    requested_aggregation_levels = [ViewAggregationLevel.SYSTEM]  # Always generate the system views
    for agg_level, requested in [
        (ViewAggregationLevel.SOCKET, args.socket_view),
        (ViewAggregationLevel.CORE, args.core_view),
        (ViewAggregationLevel.THREAD, args.thread_view),
    ]:
        if requested:
            requested_aggregation_levels.append(agg_level)
    return requested_aggregation_levels


def _append_uncore_devices(devices: List[Device],
                           valid_device_names: List[str],
                           parser: EmonParser,
                           metric_computer: MetricComputer) -> List[Device]:
    uncore_devices = parser.event_info.loc[parser.event_info[redc.DEVICE].str.startswith('UNC_'),
                                           redc.DEVICE].unique()
    valid_device_names.extend(uncore_devices)
    Device.set_valid_device_names(valid_device_names)
    for device_name in uncore_devices:
        devices.append(Device(device_name, [ViewAggregationLevel.UNCORE], metric_computer))
    return devices


def _initialize_views(args, emon_parser, devices: List[Device]) -> ViewCollection:
    normalizer = Normalizer(emon_parser.system_info.ref_tsc)
    view_collection = ViewCollection()
    for device in devices:
        for agg_level in device.aggregation_levels:
            view_name_template = f"__edp{device.decorate_label(prefix='_')}_{agg_level.name.lower()}_view_{{type}}"
            view_collection.add_view(view_name_template.format(type=ViewType.SUMMARY.name.lower()),
                                     ViewType.SUMMARY, agg_level, device, emon_parser.system_info.has_modules,
                                     device.metric_computer, normalizer, emon_parser.event_info)
            detail_views_requested = not args.no_detail_views
            if detail_views_requested:
                view_collection.add_view(view_name_template.format(type=ViewType.DETAILS.name.lower()),
                                         ViewType.DETAILS, agg_level, device, emon_parser.system_info.has_modules,
                                         device.metric_computer, normalizer, emon_parser.event_info)

    return view_collection


def _error(msg, exit_code=1, exception=None):
    """
    Print an error and terminate the script
    :param msg: error message to print
    :param exit_code: optional exit code (default is 1)
    :param exception: optional exception object
    """
    if exception:
        msg += f': {exception}'
    print(msg, file=sys.stderr)
    exit(exit_code)


def _print_banner():
    print(f'EDP version {__version__}')
    print(f'Python {sys.version}')
