#include <stdio.h>
#include <string.h>
#include <arpa/inet.h>
#include <Python.h>
#include <stdlib.h>

#include "type_interact.h"


/* Just an array for easy access and change of the size of the types.
   Must be in the same order than the values of rd_base_types
 */
size_t rd_base_types_sizes[] = {
	sizeof(char*), sizeof(double), sizeof(uint32_t), sizeof(uint16_t), sizeof(uint64_t), sizeof(uint8_t)
};

struct rd_type types[RD_TYPES_COUNT];

const char* type_list_string()
{
	static char type_list[300];
	size_t written = 0;
	size_t i = 0;

	if (i < RD_TYPES_COUNT) {
		written += snprintf(type_list + written, 300 - written, "%s", types[i].name);

		for (i = 1; i < RD_TYPES_COUNT; i++) {
			written += snprintf(type_list + written, 300 - written, ", %s", types[i].name);
		}
	}

	return type_list;
}

struct rd_type* find_type(const char* type_name)
{
	size_t i;

	for (i = 0; i < RD_TYPES_COUNT; i++) {
		if (strncmp(type_name, types[i].name, MAX_TYPE_NAME) == 0) {
			return types + i;
		}
	}

	return NULL;
}

/** Functions for reading from words. */

static void read_ip(const char* word, uint32_t* dest)
{
	struct in_addr ip_addr;
	inet_pton(AF_INET, word, &ip_addr); // Ignore return value, silent errors.

	*dest = ntohl(ip_addr.s_addr);
}

static void read_int(const char* word, uint32_t* dest)
{
	*dest = (uint32_t) strtol(word, NULL, 10);
}

static void read_long(const char* word, uint64_t* dest)
{
	*dest = (uint64_t) strtol(word, NULL, 10);
}

static void read_short(const char* word, uint16_t* dest)
{
	*dest = (uint16_t) strtol(word, NULL, 10);
}

static void read_byte(const char* word, uint8_t* dest)
{
	*dest = (uint8_t) strtol(word, NULL, 10);
}

static void read_double(const char* word, double* dest)
{
	*dest = strtod(word, NULL);
}

static void read_unixtstamp(const char* word, uint64_t* dest)
{
	char *ptr;
	uint64_t base = SEC_TO_NSEC;

	/* NOTE: negative timestamps are converted to -1 */
	if (*word == '-') {
		*dest = -1;
		return ;
	}

#ifdef DEBUG
	errno = 0;
#endif

	*dest = strtol(word, &ptr, 10);

#ifdef DEBUG
	if (errno == ERANGE) {
		fprintf(stderr, "ccore WARNING: overflow parsing timestamp (%.*s)!\n", (int)(ptr-word + 1), word);
	}
#endif

	/*
	 A unix timestamp consists in a number from 0 to 2**32-1, so it
	 can be stored in 32 bits. A unix timestamp in nanoseconds,
	 has upto 20 digits, so it can be stored in 64 bits.
	 */

	*dest *= base;
	base = SEC_TO_NSEC / 10;
	/* ptr should point to the dot, skip it!
	   if it points to something else, stop parsing
	   note: we can't use strtol here because
	         the number might not be 0 padded
	 */
	if (*ptr == '.') {
		while (isdigit(*++ptr) && base > 0) {
			*dest += (uint64_t)(*ptr - '0') * base;
			base /= 10;
		}
	}
}

// Fast conversion hex -> int
// https://stackoverflow.com/a/34573398/783010
inline uint64_t htoi(int x) {
    return 9 * (x >> 6) + (x & 017);
}

uint64_t mac_addr_to_num_htoi(const char* mac) {
	uint64_t result = 0;
	size_t shift = 44;
	size_t i;
	for (i = 0; i < 17 && mac[i] != '\0'; i++) {
		if (mac[i] == ':')
			continue;

		result += htoi(mac[i]) << shift;
		shift -= 4;
	}

	return result;
}

uint64_t mac_addr_to_num_sscanf(const char* mac) {
	unsigned char a[6];
	int last = -1;
	sscanf(mac, "%hhx:%hhx:%hhx:%hhx:%hhx:%hhx%n",
		a + 0, a + 1, a + 2, a + 3, a + 4, a + 5,
		&last);

	return
		((uint64_t)a[0]) << 40 |
		((uint64_t)a[1]) << 32 |
		((uint64_t)a[2]) << 24 |
		((uint64_t)a[3]) << 16 |
		((uint64_t)a[4]) << 8 |
		((uint64_t)a[5]);
}

static void read_mac(const char* word, uint64_t* dest)
{
	*dest = mac_addr_to_num_htoi(word); // htoi is faster
}

/** Functions to convert to Python objects */

static PyObject* python_int(uint32_t* src, size_t __attribute__((__unused__)) size)
{
	return PyInt_FromLong((long) *src);
}

static PyObject* python_long(uint64_t* src, size_t __attribute__((__unused__)) size)
{
	return PyInt_FromLong((long) *src);
}

static PyObject* python_short(uint16_t* src, size_t __attribute__((__unused__)) size)
{
	return PyInt_FromLong((long) *src);
}

static PyObject* python_byte(uint8_t* src, size_t __attribute__((__unused__)) size)
{
	return PyInt_FromLong((long) *src);
}

static PyObject* python_double(double* src, size_t __attribute__((__unused__)) size)
{
	return PyFloat_FromDouble(*src);
}

static PyObject* python_bool(uint8_t* src, size_t __attribute__((__unused__)) size)
{
	return PyBool_FromLong((long) * src);
}

static PyObject* python_str(char** src, size_t size)
{
	if (size == 0) {
		return PyString_FromString(""); // Use an empty string to avoid problems
	} else {
		return PyString_FromStringAndSize(*src, size);
	}
}

/** Functions to convert from python objects */

static size_t c_int(PyObject* src, uint32_t* dst)
{
	*dst = PyLong_AsLong(src);

	return sizeof(uint32_t);
}
static size_t c_long(PyObject* src, uint64_t* dst)
{
	*dst = PyLong_AsLong(src);

	return sizeof(uint64_t);
}
static size_t c_short(PyObject* src, uint16_t* dst)
{
	*dst = PyLong_AsLong(src);

	return sizeof(uint16_t);
}

static size_t c_byte(PyObject* src, uint8_t* dst)
{
	*dst = PyLong_AsLong(src);

	return sizeof(uint8_t);
}

static size_t c_double(PyObject* src, double* dst)
{
	*dst = PyFloat_AsDouble(src);

	return sizeof(double);
}

static size_t c_bool(PyObject* src, uint8_t* dst)
{
	*dst = PyObject_IsTrue(src);

	return sizeof(uint8_t);
}

void init_type_interact(void)
{
	// Types with variable size do not need to provide a read function.
	add_type(STRING, T_STRING, (type_read_str) NULL, (type_to_pyobject) python_str, 1, "string", (type_from_pyobject) NULL);

	add_type(DOUBLE, T_DOUBLE, (type_read_str) read_double, (type_to_pyobject) python_double, 0, "double", (type_from_pyobject) c_double);
	add_type(INT, T_INT, (type_read_str) read_int, (type_to_pyobject) python_int, 0, "int", (type_from_pyobject) c_int);
	add_type(SHORT, T_SHORT, (type_read_str) read_short, (type_to_pyobject) python_short, 0, "short", (type_from_pyobject) c_short);
	add_type(LONG, T_LONG, (type_read_str) read_long, (type_to_pyobject) python_long, 0, "long", (type_from_pyobject) c_long);
	add_type(BYTE, T_BYTE, (type_read_str) read_byte, (type_to_pyobject) python_byte, 0, "byte", (type_from_pyobject) c_byte);

	// The timestamps are handled as a special case and do not need to provide a PyObject function.
	add_type(TSTAMP, T_LONG, (type_read_str) read_unixtstamp, (type_to_pyobject) NULL, 0, "tstamp", (type_from_pyobject) NULL);

	add_type(IP, T_INT, (type_read_str) read_ip, (type_to_pyobject) python_int, 0, "ip", (type_from_pyobject) c_int);
	add_type(BOOL, T_BYTE, (type_read_str) read_byte, (type_to_pyobject) python_bool, 0, "bool", (type_from_pyobject) c_bool);
	add_type(MAC, T_LONG, (type_read_str) read_mac, (type_to_pyobject) python_long, 0, "mac", (type_from_pyobject) c_long);
}


enum NPY_TYPES storage_type_to_numpy(rd_base_types type)
{
	switch (type) {
		case T_BYTE:
			return NPY_INT8;

		case T_DOUBLE:
			return NPY_DOUBLE;

		case T_INT:
			return NPY_INT32;

		case T_LONG:
			return NPY_INT64;

		case T_SHORT:
			return NPY_INT16;

		default:
			return NPY_STRING;
	}
}

size_t rdtype_storage_size(const struct rd_type* tp)
{
	return rd_base_types_sizes[tp->storage_type];
}
