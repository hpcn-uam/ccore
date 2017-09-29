# -*- coding: utf-8 -*-

import pytest
import pandas as pd
import py
import random

@pytest.fixture(scope='session')
def sessiondir(request):
	"""Temporal session directory"""
	d = py.path.local.mkdtemp()
	request.addfinalizer(lambda: d.remove(rec=1))

	return d

@pytest.fixture(scope = 'session')
def sample_values():
	"""Returns a dictionary with sample values used in file generators,
	so the tests know whether the generated records are correct."""
	values = {
		'tz': 'America/Guatemala',
		'ip1': '192.168.1.1',
		'ip2': '10.2.2.2',
		'ip3': '192.168.1.2',
		'mac1': '6e:40:08:79:76:00',
		'mac2': '6e:40:08:79:76:01',
		'bytes': 81279,
		'frames': 129378,
		'ts_start': pd.Timestamp('2010-01-01 00:00'),
		'ts_end': pd.Timestamp('2010-01-01 00:01'),
		'tcp_frames': 1928738,
		'tcp_bytes': 18718,
		'tcp_ports': '22,80',
		'num_records': 2000,
		'frame_id': 123,
		'vlan1_id': 1,
		'vlan1_name': 'testvlan1',
		'vlan1_net': '172.16.10.0',
		'vlan2_id': 2,
		'vlan2_name': 'testvlan2',
		'vlan2_net': '172.16.20.0',
		'num_vlans': 2,
		'ip_in_vlan1': '172.16.10.65',
		'tseries_duration_sec': 60,
		'port1': 80,
		'port2': 1000,
		'more_MACs_int': 1,
		'more_MACs': True,
		'rt': 0.05,
		'result_msg': 'OK',
		'result_code': 200,
		'method': 'POST',
		'host': 'google',
		'url': '/search',
		'netmask': '255.255.255.0'
	}

	values['ts_start'] = values['ts_start'].tz_localize(values['tz'])
	values['ts_end'] = values['ts_end'].tz_localize(values['tz'])
	values['ts_start_epoch'] = values['ts_start'].value // 10 ** 9
	values['ts_end_epoch'] = values['ts_end'].value // 10 ** 9

	return values


@pytest.fixture(scope = 'session')
def ipconvs_file(sessiondir, sample_values):
	fout = sessiondir.join('ipconvs.dat')

	line = "{ip1} {ip2} {mac1} {mac2} 1 {frames} {bytes} {frames} {bytes}"
	line += " {ts_start_epoch} {ts_end_epoch} {tcp_frames} {tcp_bytes}"
	line += " {tcp_ports} {tcp_ports} (null) (null) 0"

	line = line.format(**sample_values)

	contents = [line] * sample_values['num_records']
	fout.write("\n".join(contents))

	return fout


@pytest.fixture(scope = 'session')
def macconvs_file(sessiondir, sample_values):
	fout = sessiondir.join('macconvs.dat')

	line = "{mac1} {mac2} {frames} {bytes} {ts_start_epoch} {ts_end_epoch} {frame_id} {frame_id}"
	line = line.format(**sample_values)

	contents = [line] * sample_values['num_records']
	fout.write("\n".join(contents))

	return fout

