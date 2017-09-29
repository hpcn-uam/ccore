/*
 * Author: Eduardo Miravalls Sierra <eduardo.miravalls@estudiante.uam.es>
 * Date:   2016/03/02
 *
 * A simple C module to read a procesaConexiones file and
 * return it's contents as a list of tuples of a selection
 * of its fields.
 */

#define _GNU_SOURCE
#include <Python.h>
#include <inttypes.h>

PyObject *pandas_dict = NULL;
PyObject *pandas_Timestamp = NULL;

static PyObject *
procesa_parser(char *buffer, PyObject *kwargs)
{
	size_t i;

	PyObject *args = NULL;

	PyObject *tuple = NULL;
	PyObject *srcIP = NULL;                      /* field number: 1 */
	PyObject *srcPort = NULL;                    /* field number: 2 */
	PyObject *dstIP = NULL;                      /* field number: 3 */
	PyObject *dstPort = NULL;                    /* field number: 4 */
	PyObject *firstPacketTime = NULL;            /* field number: 5 */
	PyObject *lastPacketTime = NULL;             /* field number: 6 */
	PyObject *firstSYNSrcToDstTime = NULL;       /* field number: 7 */
	PyObject *firstSYNDstToSrcTime = NULL;       /* field number: 8 */

	PyObject *firstDataSrcToDstTime = NULL;      /* field number: 11 */
	PyObject *firstDataDstToSrcTime = NULL;      /* field number: 12 */

	PyObject *numberPacketsSrcToDst = NULL;      /* field number: 17 */
	PyObject *numberPacketsDstToSrc = NULL;      /* field number: 18 */
	PyObject *numberSYNSrcToDst = NULL;          /* field number: 19 */
	PyObject *numberSYNDstToSrc = NULL;          /* field number: 20 */

	PyObject *numberRSTSrcToDst = NULL;          /* field number: 23 */
	PyObject *numberRSTDstToSrc = NULL;          /* field number: 24 */
	PyObject *numberPacketsDataSrcToDst = NULL;  /* field number: 25 */
	PyObject *numberPacketsDataDstToSrc = NULL;  /* field number: 26 */
	PyObject *numberDupACKsSrcToDst = NULL;      /* field number: 27 */
	PyObject *numberDupACKsDstToSrc = NULL;      /* field number: 28 */

	PyObject *bytesPhySrcToDst = NULL;           /* field number: 35 */
	PyObject *bytesPhyDstToSrc = NULL;           /* field number: 36 */

	PyObject *numberReTxSrcToDst = NULL;         /* field number: 59 */
	long iReTxSrcToDst;

	PyObject *numberReTxDstToSrc = NULL;         /* field number: 63 */
	long iReTxDstToSrc;

	PyObject *numCierresWin0SrcToDst = NULL;     /* field number: 89 */

	PyObject *numCierresWin0DstToSrc = NULL;     /* field number: 92 */

	long numberDataKeepAliveSrcToDst;            /* field number: 95 */
	long numberDataKeepAliveDstToSrc;            /* field number: 96 */

	char *cursor = buffer;
	char *space = strchrnul(cursor, (unsigned)' ');

	srcIP = PyString_FromStringAndSize(cursor, space - cursor);
	srcPort = PyInt_FromLong(strtol(space + 1, &cursor, 10));
	cursor++;
	space = strchrnul(cursor, (unsigned)' ');
	dstIP = PyString_FromStringAndSize(cursor, space - cursor);
	dstPort = PyInt_FromLong(strtol(space + 1, &cursor, 10));
	/*
	firstPacketTime = PyFloat_FromDouble(strtod(cursor + 1, &cursor));
	lastPacketTime = PyFloat_FromDouble(strtod(cursor + 1, &cursor));
	firstSYNSrcToDstTime = PyFloat_FromDouble(strtod(cursor + 1, &cursor));
	firstSYNDstToSrcTime = PyFloat_FromDouble(strtod(cursor + 1, &cursor));
	*/
	if (NULL == (args = Py_BuildValue("(d)", strtod(cursor + 1, &cursor)))) {
		goto error;
	}

	firstPacketTime = PyObject_Call(pandas_Timestamp, args, kwargs);
	Py_CLEAR(args);

	if (NULL == (args = Py_BuildValue("(d)", strtod(cursor + 1, &cursor)))) {
		goto error;
	}

	lastPacketTime = PyObject_Call(pandas_Timestamp, args, kwargs);
	Py_CLEAR(args);

	if (NULL == (args = Py_BuildValue("(d)", strtod(cursor + 1, &cursor)))) {
		goto error;
	}

	firstSYNSrcToDstTime = PyObject_Call(pandas_Timestamp, args, kwargs);
	Py_CLEAR(args);

	if (NULL == (args = Py_BuildValue("(d)", strtod(cursor + 1, &cursor)))) {
		goto error;
	}

	firstSYNDstToSrcTime = PyObject_Call(pandas_Timestamp, args, kwargs);
	Py_CLEAR(args);

	for (i = 9; i < 11; i++) {
		cursor = strchrnul(cursor + 1, (unsigned)' ');
	}
/*
	firstDataSrcToDstTime = PyFloat_FromDouble(strtod(cursor + 1, &cursor));
	firstDataDstToSrcTime = PyFloat_FromDouble(strtod(cursor + 1, &cursor));
*/
	if (NULL == (args = Py_BuildValue("(d)", strtod(cursor + 1, &cursor)))) {
		goto error;
	}

	firstDataSrcToDstTime = PyObject_Call(pandas_Timestamp, args, kwargs);
	Py_CLEAR(args);

	if (NULL == (args = Py_BuildValue("(d)", strtod(cursor + 1, &cursor)))) {
		goto error;
	}

	firstDataDstToSrcTime = PyObject_Call(pandas_Timestamp, args, kwargs);
	Py_CLEAR(args);

	for (i = 13; i < 17; i++) {
		cursor = strchrnul(cursor + 1, (unsigned)' ');
	}

	numberPacketsSrcToDst = PyInt_FromLong(strtol(cursor + 1, &cursor, 10));
	numberPacketsDstToSrc = PyInt_FromLong(strtol(cursor + 1, &cursor, 10));
	numberSYNSrcToDst = PyInt_FromLong(strtol(cursor + 1, &cursor, 10));
	numberSYNDstToSrc = PyInt_FromLong(strtol(cursor + 1, &cursor, 10));

	for (i = 21; i < 23; i++) {
		cursor = strchrnul(cursor + 1, (unsigned)' ');
	}

	numberRSTSrcToDst = PyInt_FromLong(strtol(cursor + 1, &cursor, 10));
	numberRSTDstToSrc = PyInt_FromLong(strtol(cursor + 1, &cursor, 10));
	numberPacketsDataSrcToDst = PyInt_FromLong(strtol(cursor + 1, &cursor, 10));
	numberPacketsDataDstToSrc = PyInt_FromLong(strtol(cursor + 1, &cursor, 10));
	numberDupACKsSrcToDst = PyInt_FromLong(strtol(cursor + 1, &cursor, 10));
	numberDupACKsDstToSrc = PyInt_FromLong(strtol(cursor + 1, &cursor, 10));

	for (i = 29; i < 35; i++) {
		cursor = strchrnul(cursor + 1, (unsigned)' ');
	}

	bytesPhySrcToDst = PyInt_FromLong(strtol(cursor + 1, &cursor, 10));
	bytesPhyDstToSrc = PyInt_FromLong(strtol(cursor + 1, &cursor, 10));

	for (i = 37; i < 59; i++) {
		cursor = strchrnul(cursor + 1, (unsigned)' ');
	}

	iReTxSrcToDst = strtol(cursor + 1, &cursor, 10);

	for (i = 60; i < 63; i++) {
		cursor = strchrnul(cursor + 1, (unsigned)' ');
	}

	iReTxDstToSrc = strtol(cursor + 1, &cursor, 10);

	for (i = 64; i < 89; i++) {
		cursor = strchrnul(cursor + 1, (unsigned)' ');
	}

	numCierresWin0SrcToDst = PyInt_FromLong(strtol(cursor + 1, &cursor, 10));

	for (i = 90; i < 92; i++) {
		cursor = strchrnul(cursor + 1, (unsigned)' ');
	}

	numCierresWin0DstToSrc = PyInt_FromLong(strtol(cursor + 1, &cursor, 10));

	for (i = 93; i < 95; i++) {
		cursor = strchrnul(cursor + 1, (unsigned)' ');
	}

	numberDataKeepAliveSrcToDst = strtol(cursor + 1, &cursor, 10);
	numberDataKeepAliveDstToSrc = strtol(cursor + 1, &cursor, 10);

	if (numberDataKeepAliveSrcToDst < 0) {
		numberDataKeepAliveSrcToDst = 0;
	}

	if (numberDataKeepAliveDstToSrc < 0) {
		numberDataKeepAliveDstToSrc = 0;
	}

	numberReTxSrcToDst = PyInt_FromLong(iReTxSrcToDst - numberDataKeepAliveSrcToDst);
	numberReTxDstToSrc = PyInt_FromLong(iReTxDstToSrc - numberDataKeepAliveDstToSrc);

	if (NULL == srcIP ||
	        NULL == srcPort ||
	        NULL == dstIP ||
	        NULL == dstPort ||
	        NULL == firstPacketTime ||
	        NULL == lastPacketTime ||
	        NULL == firstSYNSrcToDstTime ||
	        NULL == firstSYNDstToSrcTime ||
	        NULL == firstDataSrcToDstTime ||
	        NULL == firstDataDstToSrcTime ||
	        NULL == numberPacketsSrcToDst ||
	        NULL == numberPacketsDstToSrc ||
	        NULL == numberSYNSrcToDst ||
	        NULL == numberSYNDstToSrc ||
	        NULL == numberRSTSrcToDst ||
	        NULL == numberRSTDstToSrc ||
	        NULL == numberPacketsDataSrcToDst ||
	        NULL == numberPacketsDataDstToSrc ||
	        NULL == numberDupACKsSrcToDst ||
	        NULL == numberDupACKsDstToSrc ||
	        NULL == bytesPhySrcToDst ||
	        NULL == bytesPhyDstToSrc ||
	        NULL == numberReTxSrcToDst ||
	        NULL == numberReTxDstToSrc ||
	        NULL == numCierresWin0SrcToDst ||
	        NULL == numCierresWin0DstToSrc) {
error:
		Py_CLEAR(srcIP);
		Py_CLEAR(srcPort);
		Py_CLEAR(dstIP);
		Py_CLEAR(dstPort);
		Py_CLEAR(firstPacketTime);
		Py_CLEAR(lastPacketTime);
		Py_CLEAR(firstSYNSrcToDstTime);
		Py_CLEAR(firstSYNDstToSrcTime);
		Py_CLEAR(firstDataSrcToDstTime);
		Py_CLEAR(firstDataDstToSrcTime);
		Py_CLEAR(numberPacketsSrcToDst);
		Py_CLEAR(numberPacketsDstToSrc);
		Py_CLEAR(numberSYNSrcToDst);
		Py_CLEAR(numberSYNDstToSrc);
		Py_CLEAR(numberRSTSrcToDst);
		Py_CLEAR(numberRSTDstToSrc);
		Py_CLEAR(numberPacketsDataSrcToDst);
		Py_CLEAR(numberPacketsDataDstToSrc);
		Py_CLEAR(numberDupACKsSrcToDst);
		Py_CLEAR(numberDupACKsDstToSrc);
		Py_CLEAR(bytesPhySrcToDst);
		Py_CLEAR(bytesPhyDstToSrc);
		Py_CLEAR(numberReTxSrcToDst);
		Py_CLEAR(numberReTxDstToSrc);
		Py_CLEAR(numCierresWin0SrcToDst);
		Py_CLEAR(numCierresWin0DstToSrc);
		return NULL;
	}

	if (NULL != (tuple = Py_BuildValue("(OOOOOOOOOOOOOOOOOOOOOOOOOO)",
	                                   srcIP,
	                                   srcPort,
	                                   dstIP,
	                                   dstPort,
	                                   firstPacketTime,
	                                   lastPacketTime,
	                                   firstSYNSrcToDstTime,
	                                   firstSYNDstToSrcTime,
	                                   firstDataSrcToDstTime,
	                                   firstDataDstToSrcTime,
	                                   numberPacketsSrcToDst,
	                                   numberPacketsDstToSrc,
	                                   numberSYNSrcToDst,
	                                   numberSYNDstToSrc,
	                                   numberRSTSrcToDst,
	                                   numberRSTDstToSrc,
	                                   numberPacketsDataSrcToDst,
	                                   numberPacketsDataDstToSrc,
	                                   numberDupACKsSrcToDst,
	                                   numberDupACKsDstToSrc,
	                                   bytesPhySrcToDst,
	                                   bytesPhyDstToSrc,
	                                   numberReTxSrcToDst,
	                                   numberReTxDstToSrc,
	                                   numCierresWin0SrcToDst,
	                                   numCierresWin0DstToSrc))) {
		Py_CLEAR(srcIP);
		Py_CLEAR(srcPort);
		Py_CLEAR(dstIP);
		Py_CLEAR(dstPort);
		Py_CLEAR(firstPacketTime);
		Py_CLEAR(lastPacketTime);
		Py_CLEAR(firstSYNSrcToDstTime);
		Py_CLEAR(firstSYNDstToSrcTime);
		Py_CLEAR(firstDataSrcToDstTime);
		Py_CLEAR(firstDataDstToSrcTime);
		Py_CLEAR(numberPacketsSrcToDst);
		Py_CLEAR(numberPacketsDstToSrc);
		Py_CLEAR(numberSYNSrcToDst);
		Py_CLEAR(numberSYNDstToSrc);
		Py_CLEAR(numberRSTSrcToDst);
		Py_CLEAR(numberRSTDstToSrc);
		Py_CLEAR(numberPacketsDataSrcToDst);
		Py_CLEAR(numberPacketsDataDstToSrc);
		Py_CLEAR(numberDupACKsSrcToDst);
		Py_CLEAR(numberDupACKsDstToSrc);
		Py_CLEAR(bytesPhySrcToDst);
		Py_CLEAR(bytesPhyDstToSrc);
		Py_CLEAR(numberReTxSrcToDst);
		Py_CLEAR(numberReTxDstToSrc);
		Py_CLEAR(numCierresWin0SrcToDst);
		Py_CLEAR(numCierresWin0DstToSrc);
		return tuple;

	} else {
		return NULL;
	}
}


static PyObject *
mockparser_fastReader(PyObject *self, PyObject *args)
{
	const char *path = NULL;
	size_t buffer_size = 1024 * 1024; /* 1MB */
	size_t initial_size = 0;
	char *buffer = NULL;
	FILE *f = NULL;

	PyObject *tuple = NULL;
	PyObject *list = NULL;
	PyObject *tz_in = NULL;
	PyObject *kwargs = NULL;

	if (!PyArg_ParseTuple(args, "sO", &path, &tz_in)) {
		return NULL;
	}

	kwargs = Py_BuildValue("{s:O,s:s}", "tz", tz_in, "unit", "s");

	if (NULL == kwargs) {
		goto error;
	}

	if (NULL == (buffer = calloc(buffer_size, 1))) {
		return PyErr_NoMemory();
	}

	if (NULL == (f = fopen(path, "r"))) {
		free(buffer);
		return PyErr_SetFromErrno(PyExc_IOError);
	}

	if (NULL == (list = PyList_New(initial_size))) {
		goto error;
	}

	while (NULL != fgets(buffer, buffer_size, f)) {
		tuple = procesa_parser(buffer, kwargs);

		if (NULL == tuple || 0 != PyList_Append(list, tuple)) {
			goto error;
		}

		Py_CLEAR(tuple);
	}


	Py_CLEAR(kwargs);
	Py_CLEAR(tz_in);
	free(buffer);
	fclose(f);
	return list;

error:
	Py_CLEAR(list);
	Py_CLEAR(tuple);
	Py_CLEAR(kwargs);
	Py_CLEAR(tz_in);
	free(buffer);
	fclose(f);
	return NULL;
}



static PyMethodDef MockParserMethods[] = {
	{	"read",  mockparser_fastReader, METH_VARARGS,
		"Read a file."
	},
	{NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initmockparser(void)
{
	PyObject *m;
	PyObject *pandas_module;

	m = Py_InitModule("mockparser", MockParserMethods);
	if (m == NULL) {
		return;
	}

	/* load pandas module */
	if (NULL == (pandas_module = PyImport_ImportModule("pandas"))) {
		return ;
	}

	/* get pandas __dict__ */
	pandas_dict = PyModule_GetDict(pandas_module);

	/* obtain pandas Timestamp constructor */
	pandas_Timestamp = PyDict_GetItemString(pandas_dict, "Timestamp");
}
