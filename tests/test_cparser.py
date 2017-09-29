# -*- coding: utf-8 -*-

import pytest
import cparser
import pandas as pd
import pytz
from utils import ip2int

class Test_RecordList(object):
	def test_new(self):
		parser = cparser.RecordList()

		assert parser is not None

	def test_read___gets_all_records(self, macconvs_file, sample_values):
		parser = cparser.RecordList()

		params = {
			'srcMAC': (0, 'string'),
			'dstMAC': (1, 'string'),
			'frames': (2, 'int'),
			'bytes': (3, 'int'),
			'ts_start': (4, 'double'),
			'ts_end': (5, 'double'),
		}

		parser.read(macconvs_file.strpath, params)

		assert len(parser) == sample_values['num_records']

	def test_read__records_are_correct(self, macconvs_file, sample_values):
		parser = cparser.RecordList()

		params = {
			'srcMAC': (0, 'string'),
			'dstMAC': (1, 'string'),
			'frames': (2, 'int'),
			'bytes': (3, 'int'),
			'ts_start': (4, 'double'),
			'ts_end': (5, 'double'),
		}

		parser.read(macconvs_file.strpath, params)

		record = parser[0]

		assert record.srcMAC == sample_values['mac1']
		assert record.dstMAC == sample_values['mac2']
		assert record.frames == sample_values['frames']
		assert record.bytes == sample_values['bytes']
		assert record.ts_start == sample_values['ts_start_epoch']
		assert record.ts_end == sample_values['ts_end_epoch']

	def test_iter__can_iterate(self,  macconvs_file, sample_values):
		parser = cparser.RecordList()

		params = {
			'srcMAC': (0, 'string'),
			'dstMAC': (1, 'string'),
			'frames': (2, 'int'),
			'bytes': (3, 'int'),
			'ts_start': (4, 'double'),
			'ts_end': (5, 'double'),
		}

		parser.read(macconvs_file.strpath, params)
		num_iterated = 0

		for record in parser:
			assert record.srcMAC == sample_values['mac1']
			assert record.dstMAC == sample_values['mac2']
			assert record.frames == sample_values['frames']
			assert record.bytes == sample_values['bytes']
			assert record.ts_start == sample_values['ts_start_epoch']
			assert record.ts_end == sample_values['ts_end_epoch']
			num_iterated += 1

		assert num_iterated == sample_values['num_records']

	def test_read__can_read_IPs(self, ipconvs_file, sample_values):
		parser = cparser.RecordList()

		params = {
			'srcIP': (0, 'ip')
		}

		parser.read(ipconvs_file.strpath, params)

		assert len(parser) > 0
		assert parser[0].srcIP == ip2int(sample_values['ip1'])

	def test_read__can_read_pandas_Timestamps(self, macconvs_file, sample_values):
		parser = cparser.RecordList(sample_values['tz'])

		params = {
			'ts_start': (4, 'tstamp'),
		}

		parser.read(macconvs_file.strpath, params)

		assert len(parser) > 0

		tstamp = parser[0].ts_start

		assert isinstance(tstamp, pd.Timestamp)
		assert tstamp == sample_values['ts_start']

	def test_read__can_read_pandas_Timestamps__timezone_is_correct(self, macconvs_file, sample_values):
		parser = cparser.RecordList('Canada/East-Saskatchewan')  # Because why not

		params = {
			'ts_start': (4, 'tstamp'),
		}

		parser.read(macconvs_file.strpath, params)

		assert len(parser) > 0

		tstamp = parser[0].ts_start

		assert isinstance(tstamp, pd.Timestamp)
		assert tstamp == sample_values['ts_start']
		assert tstamp.tz.zone == 'Canada/East-Saskatchewan'

	def test_read__negative_timestamps__ignored(self, sessiondir):
		fout = sessiondir.join('negtest.dat')

		fout.write("-1\n")

		params = {
			'ts_start': (0, 'tstamp'),
		}

		parser = cparser.RecordList()
		parser.read(fout.strpath, params)

		assert len(parser) > 0

		tstamp = parser[0].ts_start

		assert isinstance(tstamp, pd.Timestamp)
		assert tstamp.value == 0

	def test_read__negative_timestamps_fallback_available__falls_back(self, sessiondir):
		fout = sessiondir.join('fallbacktest.dat')

		fout.write("232131 -1\n")

		params = {
			'ts_start': (0, 'tstamp'),
			'other_ts': (1, 'tstamp'),
		}

		parser = cparser.RecordList()
		parser.read(fout.strpath, params)

		assert len(parser) > 0

		tstamp = parser[0].ts_start

		assert isinstance(tstamp, pd.Timestamp)
		assert tstamp.value == 232131000000000

		tstamp = parser[0].other_ts

		assert isinstance(tstamp, pd.Timestamp)
		assert tstamp.value == 232131000000000

	def test_read__stores_max_min_tstamps(self, macconvs_file, sample_values):
		parser = cparser.RecordList(sample_values['tz'])

		params = {
			'srcMAC': (0, 'string'),
			'dstMAC': (1, 'string'),
			'frames': (2, 'int'),
			'bytes': (3, 'int'),
			'ts_start': (4, 'tstamp'),
			'ts_end': (5, 'tstamp'),
		}

		parser.read(macconvs_file.strpath, params)

		assert len(parser) == sample_values['num_records']
		assert parser.min_timestamp == sample_values['ts_start']
		assert parser.max_timestamp == sample_values['ts_end']

	def test_read__no_stamps_in_file__min_max_tstamps_are_valid(self, macconvs_file, sample_values):
		parser = cparser.RecordList()

		params = {
			'srcMAC': (0, 'string'),
			'dstMAC': (1, 'string'),
			'frames': (2, 'int'),
			'bytes': (3, 'int'),
			'ts_start': (4, 'double'),
			'ts_end': (5, 'double'),
		}

		parser.read(macconvs_file.strpath, params)

		assert len(parser) == sample_values['num_records']
		assert parser.min_timestamp == pd.Timestamp(0)
		assert parser.max_timestamp == pd.Timestamp(0)

	def test_read__negative_zero_timestamps__ignored_min_max(self, sessiondir):
		fout = sessiondir.join('negtest.dat')

		lines = ["-1", "0", "200"]

		fout.write("\n".join(lines))

		params = {
			'ts_start': (0, 'tstamp'),
		}

		parser = cparser.RecordList()
		parser.read(fout.strpath, params)

		assert len(parser) == 3
		assert parser.min_timestamp == pd.Timestamp(200, unit = 's')
		assert parser.max_timestamp == pd.Timestamp(200, unit = 's')

	def test_read__bool_as_c_integers__parsed_as_bools(self, sessiondir):
		fout = sessiondir.join('booltest.dat')

		lines = ["-1", "0", "1"]

		fout.write("\n".join(lines))

		params = {
			'test': (0, 'bool'),
		}

		parser = cparser.RecordList()
		parser.read(fout.strpath, params)

		assert len(parser) == 3
		assert isinstance(parser[0].test, bool)
		assert parser[0].test is True
		assert parser[1].test is False
		assert parser[2].test is True

	def test_read__big_number__does_not_overflow(self, sessiondir):
		fout = sessiondir.join('booltest.dat')

		num = 21474836470
		lines = [str(num)]

		fout.write("\n".join(lines))

		params = {
			'test': (0, 'long'),
		}

		parser = cparser.RecordList()
		parser.read(fout.strpath, params)

		assert len(parser) == 1
		assert parser[0].test == num

	def test_read__shorts__can_read_shorts(self, sessiondir):
		fout = sessiondir.join('booltest.dat')

		num_1 = 80
		num_2 = 1238791
		lines = [str(num_1), str(num_2)]

		fout.write("\n".join(lines))

		params = {
			'test': (0, 'short'),
		}

		parser = cparser.RecordList()
		parser.read(fout.strpath, params)

		assert len(parser) == 2
		assert parser[0].test == num_1
		assert parser[1].test == num_2 % 65536

	def test_read__can_build_values(self,  macconvs_file, sample_values):
		parser = cparser.RecordList()

		params = {
			'srcMAC': (0, 'string'),
			'dstMAC': (1, 'string'),
			'frames': (2, 'int'),
			'bytes': (3, 'int'),
			'doubles': (3, 'int', lambda x: 2 * int(x)),
			'ts_start': (4, 'double'),
			'ts_end': (5, 'double'),
		}

		parser.read(macconvs_file.strpath, params)

		assert len(parser) == sample_values['num_records']

		record = parser[0]

		assert record.doubles == 2 * sample_values['bytes']

	def test_read__string_end_of_line__newline_discarded(self, sessiondir):
		fout = sessiondir.join('booltest.dat')

		lines = ["test", "test"]

		fout.write("\n".join(lines))

		params = {
			'test': (0, 'string'),
		}

		parser = cparser.RecordList()
		parser.read(fout.strpath, params)

		assert len(parser) == 2
		assert parser[0].test == "test"

	def test_read__different_separator__reads_correct_values(self, sessiondir):
		fout = sessiondir.join('separatortest.dat')

		num_1 = 80
		num_2 = 90
		line = "{0}|{1}".format(num_1, num_2)
		lines = [line] * 3

		fout.write("\n".join(lines))

		params = {
			'num1': (0, 'short'),
			'num2': (1, 'short')
		}

		parser = cparser.RecordList(pytz.timezone('UTC'), "|")
		parser.read(fout.strpath, params)

		assert len(parser) == len(lines)

		for record in parser:
			assert record.num1 == num_1
			assert record.num2 == num_2

	def test_tstamp_range__only_iterates_in_range(self, sessiondir):
		fout = sessiondir.join('rangetest.dat')

		lines = [str(n) for n in range(1000)]
		fout.write("\n".join(lines))

		params = {
			'test': (0, 'tstamp')
		}

		parser = cparser.RecordList()
		parser.read(fout.strpath, params)

		assert len(parser) == 1000

		rng = parser.tstamp_range(100*10**9, 400*10**9, ['test'])
		count = 0

		for r in rng:
			count += 1
			assert 100 <= r.test.value // 10 ** 9 <= 400

		assert count == 400 - 100 + 1

	def test_field_as_numpy__double__creates_a_numpy_array(self, macconvs_file, sample_values):
		import numpy as np

		parser = cparser.RecordList()

		params = {
			'srcMAC': (0, 'string'),
			'dstMAC': (1, 'string'),
			'frames': (2, 'int'),
			'bytes': (3, 'int'),
			'ts_start': (4, 'double'),
			'ts_end': (5, 'double'),
		}

		parser.read(macconvs_file.strpath, params)

		assert len(parser) == sample_values['num_records']

		nparr = parser.field_as_numpy('ts_start')

		assert isinstance(nparr, np.ndarray)
		assert len(nparr) == len(parser)

		for i in range(len(nparr)):
			from_nparr = nparr[i]
			from_parser = parser[i].ts_start

			assert from_nparr == from_parser

	def test_field_as_numpy__double__creates_a_numpy_array(self, macconvs_file, sample_values):
		import numpy as np

		parser = cparser.RecordList()

		params = {
			'srcMAC': (0, 'string'),
			'dstMAC': (1, 'string'),
			'frames': (2, 'int'),
			'bytes': (3, 'int'),
			'ts_start': (4, 'double'),
			'ts_end': (5, 'double'),
		}

		parser.read(macconvs_file.strpath, params)

		assert len(parser) == sample_values['num_records']

		nparr = parser.field_as_numpy('frames')

		assert isinstance(nparr, np.ndarray)
		assert len(nparr) == len(parser)

		for i in range(len(nparr)):
			from_nparr = nparr[i]
			from_parser = parser[i].frames

			assert from_nparr == from_parser

	def test_field_as_numpy__tstamp__creates_a_numpy_array(self, macconvs_file, sample_values):
		import numpy as np

		parser = cparser.RecordList()

		params = {
			'srcMAC': (0, 'string'),
			'dstMAC': (1, 'string'),
			'frames': (2, 'int'),
			'bytes': (3, 'int'),
			'ts_start': (4, 'tstamp'),
			'ts_end': (5, 'double'),
		}

		parser.read(macconvs_file.strpath, params)

		assert len(parser) == sample_values['num_records']

		nparr = parser.field_as_numpy('ts_start')

		assert isinstance(nparr, np.ndarray)
		assert len(nparr) == len(parser)

		for i in range(len(nparr)):
			from_nparr = nparr[i]
			from_parser = parser[i].ts_start

			assert from_nparr == from_parser.to_datetime64()

	def test_field_as_numpy__tstamp_defaut_value__creates_a_numpy_array(self, sessiondir):
		import pandas as pd
		import numpy as np

		fout = sessiondir.join('deftstampvals.dat')

		tstamps = [pd.Timestamp('2000-10-10'), pd.Timestamp('2000-10-11')]
		tstamps_str = ["0", str(pd.Timestamp('2000-10-11').value // 10 ** 9)]

		fout.write("\n".join(tstamps_str))

		params = {
			'test': (0, 'tstamp')
		}

		parser = cparser.RecordList()
		parser.read(fout.strpath, params)

		assert len(parser) == 2

		nparr = parser.field_as_numpy('test', tstamps[0].value)

		assert isinstance(nparr, np.ndarray)
		assert len(nparr) == len(parser)

		for i in range(len(parser)):
			from_nparr = nparr[i]
			expected = tstamps[i]
			assert from_nparr == expected.to_datetime64()

	def test_filter_fields__only_iterates_in_range(self, sessiondir):
		fout = sessiondir.join('filter.dat')

		lines = [str(n) + " " + str(n % 2) for n in range(1000)]
		fout.write("\n".join(lines))

		params = {
			'test': (0, 'tstamp'),
			'test2': (1, 'int')
		}

		parser = cparser.RecordList()
		parser.read(fout.strpath, params)

		assert len(parser) == 1000

		rng = parser.filter_fields(1, ['test2'])
		count = 0

		for r in rng:
			count += 1
			assert r.test2 == 1

		assert count == 1000 / 2

	def test_tstamp_range_and_filter__only_iterates_in_range(self, sessiondir):
		fout = sessiondir.join('combinediter.dat')

		lines = [str(n) + " " + str(n % 2) for n in range(1000)]
		fout.write("\n".join(lines))

		params = {
			'test': (0, 'tstamp'),
			'test2': (1, 'int')
		}

		parser = cparser.RecordList()
		parser.read(fout.strpath, params)

		assert len(parser) == 1000

		rng = parser.tstamp_range(100*10**9, 400*10**9, ['test']).filter_fields(1, ['test2'])
		count = 0

		for r in rng:
			count += 1
			assert 100 <= r.test.value // 10 ** 9 <= 400 and r.test2 == 1

		assert count == (400 - 100 + 1) / 2

	def test_field_as_numpy__after_tstamp_range_and_filter__correct_array(self, sessiondir):
		import numpy as np

		fout = sessiondir.join('fieldafterfilters.dat')

		lines = [str(n) + " " + str(n % 2) for n in range(1000)]
		fout.write("\n".join(lines))

		params = {
			'test': (0, 'tstamp'),
			'test2': (1, 'int')
		}

		parser = cparser.RecordList()
		parser.read(fout.strpath, params)

		assert len(parser) == 1000

		rng = parser.tstamp_range(100*10**9, 400*10**9, ['test']).filter_fields(1, ['test2'])
		array = rng.field_as_numpy('test2')

		assert isinstance(array, np.ndarray)
		assert len(array) == (400 - 100 + 1) / 2
		assert all(array == 1)

	def test_filter_fields__multiple_iterators__separated_state(self, sessiondir):
		fout = sessiondir.join('multipleiter.dat')

		lines = [str(n) + " " + str(n % 2) for n in range(1000)]
		fout.write("\n".join(lines))

		params = {
			'test': (0, 'tstamp'),
			'test2': (1, 'int')
		}

		parser = cparser.RecordList()
		parser.read(fout.strpath, params)

		assert len(parser) == 1000

		rng = parser.tstamp_range(100*10**9, 400*10**9, ['test'])

		test1 = rng.filter_fields(1, ['test2'])
		test0 = rng.filter_fields(0, ['test2'])

		rng_count = 0
		for r in rng:
			rng_count += 1

		assert rng_count == 400 - 100 + 1

		test1_count = 0
		for r in test1:
			test1_count += 1
			assert r.test2 == 1

		assert test1_count == (400 - 100 + 1) / 2

		test0_count = 0
		for r in test0:
			test0_count += 1
			assert r.test2 == 0

		assert test0_count == (400 - 100 + 1) / 2 + 1

	def test_filter_fields__iterator_is_RecordListIter(self, sessiondir):
		fout = sessiondir.join('instanceiter.dat')

		lines = [str(n) + " " + str(n % 2) for n in range(1000)]
		fout.write("\n".join(lines))

		params = {
			'test': (0, 'tstamp'),
			'test2': (1, 'int')
		}

		parser = cparser.RecordList()
		parser.read(fout.strpath, params)

		assert isinstance(parser.filter_fields(1, ['test2']), cparser.RecordListIter)
		assert isinstance(parser.tstamp_range(1*10**9, 2*10**9, ['test']), cparser.RecordListIter)

	def test_filter_fields__is_filtering_field__true(self, sessiondir):
		fout = sessiondir.join('field_bool.dat')

		lines = [str(n) + " " + str(n % 2) for n in range(1000)]
		fout.write("\n".join(lines))

		params = {
			'test': (0, 'tstamp'),
			'test2': (1, 'int')
		}

		parser = cparser.RecordList()
		parser.read(fout.strpath, params)
		it = parser.filter_fields(1, ['test2'])

		assert it.is_filtering_field
		assert not it.is_filtering_range

	def test_filter_fields__is_filtering_range__true(self, sessiondir):
		fout = sessiondir.join('range_bool.dat')

		lines = [str(n) + " " + str(n % 2) for n in range(1000)]
		fout.write("\n".join(lines))

		params = {
			'test': (0, 'tstamp'),
			'test2': (1, 'int')
		}

		parser = cparser.RecordList()
		parser.read(fout.strpath, params)
		it = parser.tstamp_range(1*10**9, 2*10**9, ['test'])

		assert not it.is_filtering_field
		assert it.is_filtering_range

	def test_filter__supports_multiple_iterations(self, sessiondir):

		fout = sessiondir.join('filter.dat')

		lines = [str(n) + " " + str(n % 2) for n in range(1000)]
		fout.write("\n".join(lines))

		params = {
			'test': (0, 'tstamp'),
			'test2': (1, 'int')
		}

		parser = cparser.RecordList()
		parser.read(fout.strpath, params)

		assert len(parser) == 1000

		rng = parser.filter_fields(1, ['test2'])
		count = 0

		for r in rng:
			count += 1
			assert r.test2 == 1

		assert count == 1000 / 2

		count = 0

		for r in rng:
			count += 1
			assert r.test2 == 1

		assert count == 1000 / 2

	def test_read__empty_lines__ignored(self, sessiondir):
		fout = sessiondir.join('negtest.dat')

		lines = ["-1 0", "", "200 200"]

		fout.write("\n".join(lines))

		params = {
			'a': (0, 'int'),
			'b': (1, 'int'),
		}

		parser = cparser.RecordList()
		parser.read(fout.strpath, params)

		assert len(parser) == 2

	def test_read__empty_lines__ignored(self, sessiondir):
		fout = sessiondir.join('emptytest.dat')

		lines = ["-1 0", "", "200 200"]

		fout.write("\n".join(lines))

		params = {
			'a': (0, 'int'),
			'b': (1, 'int'),
		}

		parser = cparser.RecordList()
		parser.read(fout.strpath, params)

		assert len(parser) == 2

	def test_read__invalid_timestamp__does_not_crash(self, sessiondir):
		fout = sessiondir.join('invformat.dat')

		lines = ["asdsasd 0", "200 200"]

		fout.write("\n".join(lines))

		params = {
			'a': (0, 'tstamp'),
			'b': (1, 'string'),
		}

		parser = cparser.RecordList()
		parser.read(fout.strpath, params)

		assert len(parser) == 2

	def test_read__invalid_ip__does_not_crash(self, sessiondir):
		fout = sessiondir.join('invformat.dat')

		lines = ["asdsasd 0", "200 200"]

		fout.write("\n".join(lines))

		params = {
			'a': (0, 'ip'),
			'b': (1, 'string'),
		}

		parser = cparser.RecordList()
		parser.read(fout.strpath, params)

		assert len(parser) == 2

	def test_read__consecutive_spaces__ignored(self, sessiondir):
		fout = sessiondir.join('invformat.dat')

		lines = ["asdsasd      0", "200    300"]

		fout.write("\n".join(lines))

		params = {
			'a': (0, 'string'),
			'b': (1, 'string'),
		}

		parser = cparser.RecordList()
		parser.read(fout.strpath, params)

		assert len(parser) == 2
		assert parser[0].a == "asdsasd"
		assert parser[0].b == "0"
		assert parser[1].a == "200"
		assert parser[1].b == "300"


	def test_read__consecutive_special_char_separator__not_ignored(self, sessiondir):
		fout = sessiondir.join('invformat.dat')

		lines = ["asdsasd||0", "200||300", "200|aaa|300"]

		fout.write("\n".join(lines))

		params = {
			'a': (0, 'string'),
			'b': (1, 'string'),
			'c': (2, 'string'),
		}

		parser = cparser.RecordList(pytz.timezone('UTC'), "|")
		parser.read(fout.strpath, params)

		assert len(parser) == 3
		assert parser[0].a == "asdsasd"
		assert parser[0].b == ""
		assert parser[0].c == "0"
		assert parser[1].a == "200"
		assert parser[1].b == ""
		assert parser[1].c == "300"
		assert parser[2].a == "200"
		assert parser[2].b == "aaa"
		assert parser[2].c == "300"
