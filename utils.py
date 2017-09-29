# -*- coding: utf-8 -*-
"""This module contains small utility functions."""

import socket
import struct

# http://stackoverflow.com/a/9591005
def ip2int(ip):
	"""Convert an IP in dot-decimal notation to int.
    :param ip: string.
    """
	if not isinstance(ip, basestring):
		raise ValueError("ip must be str and is {0} instead".format(type(ip)))
	packedIP = socket.inet_aton(ip)
	return struct.unpack("!I", packedIP)[0]
