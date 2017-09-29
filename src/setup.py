from distutils.core import setup, Extension
from numpy.distutils.misc_util import get_info

module1 = Extension('mockparser', sources=['mockparser.c'])

module2 = Extension(
	'cparser',
	sources=['cparser.c', 'cparser-python.c', 'pandas_interact.c', 'type_interact.c', 'cparser-iterator.c'],
	#extra_compile_args = [ '-Wall', '-std=gnu99', '-march=native', '-Ofast'],
	extra_compile_args=['-Wall', '-std=gnu99', '-O3'],
	undef_macros=['NDEBUG'],
	**get_info("npymath"))

setup(name='hpat', version='1.0', description='C Core for the FERMIN application', ext_modules=[module1, module2])
