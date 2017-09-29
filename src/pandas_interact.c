/**
 * @brief Functions for interaction with the Pandas module (mainly timestamps).
 */

#include <Python.h>

PyObject* pandas_Timestamp;
PyObject* pandas_dict;


short cparser_pandas_init(void)
{
	PyObject* pandas_module;

	/* load pandas module */
	if (NULL == (pandas_module = PyImport_ImportModule("pandas"))) {
		return -1;
	}

	/* get pandas __dict__ */
	pandas_dict = PyModule_GetDict(pandas_module);

	/* obtain pandas Timestamp constructor */
	pandas_Timestamp = PyDict_GetItemString(pandas_dict, "Timestamp");

	return 0;
}

PyObject* cparser_pandas_tstamp(uint64_t value, PyObject* tz)
{
	PyObject* kwargs = NULL;
	PyObject* args = NULL;
	PyObject* tstamp = NULL;

	if (value == (uint64_t)-1) {
		value = 0; // Don't allow negative timestamps.
		// fprintf(stderr, "ccore: Timestamp was -1!\n");
	}

	args = Py_BuildValue("(L)", value);

	if (tz == NULL) {
		kwargs = Py_BuildValue("{s:s}", "unit", "ns");
	} else {
		kwargs = Py_BuildValue("{s:O,s:s}", "tz", tz, "unit", "ns");
	}

	if (args != NULL && kwargs != NULL) {
		tstamp = PyObject_Call(pandas_Timestamp, args, kwargs);
	}

	Py_CLEAR(kwargs);
	Py_CLEAR(args);

	return tstamp;
}
