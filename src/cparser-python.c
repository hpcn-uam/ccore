/**
 * @brief Functions for the interaction with Python.
 */

#include "cparser-python.h"
#include "cparser-iterator.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <string.h>
/** @section Allocators/destructors */

static void cparser_rl_dealloc(cparser_recordlist* self)
{
	assert(NULL != self);

	if (self->fields_ptr != NULL) {
		assert(NULL != self->lst);

		for (size_t i = 0; i < self->lst->num_fields; i++) {
			Py_CLEAR(self->fields_ptr[i].value_builder);
		}

		free(self->fields_ptr);
		self->fields_ptr = NULL;
	}


	recordlist_delete(self->lst);
	Py_XDECREF(self->tz);

	self->ob_type->tp_free((PyObject*) self);
}

static void cparser_r_dealloc(cparser_record* self)
{
	assert(NULL != self);

	self->lst = NULL;

	Py_XDECREF(self->tz);

	self->ob_type->tp_free((PyObject*) self);
}

static PyObject *cparser_rl_new(PyTypeObject *type,
								PyObject __attribute__((__unused__)) *args, PyObject  __attribute__((__unused__)) *kwds)
{
	cparser_recordlist *self;

	assert(NULL != type);

	self = (cparser_recordlist*) type->tp_alloc(type, 0);

	if (self != NULL) {
		self->lst = NULL;
		self->fields_ptr = NULL;
	}

	return (PyObject *)self;
}

// https://bugs.python.org/issue17380
int
cparser_rl_init(PyObject* self, PyObject* args, PyObject*  __attribute__((__unused__)) kwargs)
{
	cparser_recordlist *recordlist = (cparser_recordlist*) self;
	PyObject* tzone = NULL;
	const char* separator = " "; // Default separator is a space.

	if (!PyArg_ParseTuple(args, "|Os", &tzone, &separator)) {
		return -1;
	}

	strncpy(recordlist->separator, separator, 5);

	if (tzone != NULL) {
		recordlist->tz = tzone;
		Py_INCREF(recordlist->tz);
	} else {
		recordlist->tz = NULL;
	}

	return 0;
}

/** @section Reading methods */

static int compare_field_spec_by_column(const void* a, const void* b)
{
	assert(NULL != a);
	assert(NULL != b);

	const struct field_spec* fa = (const struct field_spec*) a;
	const struct field_spec* fb = (const struct field_spec*) b;

	return fa->column - fb->column;
}


/**
 * Read the records from a given file.
 * @param  self Python object
 * @param  args Python object with the file path (string) and a dictionary of
 *              fields to be parsed.
 * @return      None.
 */
static PyObject* cparser_rl_read(PyObject* self, PyObject* args)
{
	const char* fpath;
	PyObject* field_dict = NULL;
	PyObject *key, *value;
	size_t field_spec_idx = 0;  // I don't trust Python dict pos value.
	Py_ssize_t pos = 0;
	cparser_recordlist* recordlist = (cparser_recordlist*) self;

	assert(NULL != self);

	if (!PyArg_ParseTuple(args, "sO", &fpath, &field_dict)) {
		return NULL;
	}

	if (!PyDict_Check(field_dict)) {
		return PyErr_Format(PyExc_TypeError, "Second argument should be a dictionary");
	}

	recordlist->fields_ptr = calloc(PyDict_Size(field_dict), sizeof(struct field_spec));

	while (PyDict_Next(field_dict, &pos, &key, &value)) {
		int index;
		const char* type_str;
		const char* name;
		struct rd_type* type;
		PyObject* create_value = NULL;

		// Check that the key and value are valid
		name = PyString_AsString(key);

		if (name == NULL) {
			return PyErr_Format(PyExc_TypeError, "The dictionary values must be strings.");
		}

		if (!PyTuple_Check(value)) {
			return PyErr_Format(PyExc_TypeError, "The dictionary values must be tuples.");
		}

		if (!PyArg_ParseTuple(value, "is|O", &index, &type_str, &create_value)) {
			return PyErr_Format(PyExc_TypeError, "The tuples in the dictionary must be (int, string).");
		}

		type = find_type(type_str);

		if (type == NULL) {
			return PyErr_Format(PyExc_ValueError, "Supported types are %s. Current type is %s", type_list_string(), type_str);
		}

		if (create_value != NULL) {
			if (!PyCallable_Check(create_value)) {
				return PyErr_Format(PyExc_ValueError, "Third argument of the tuple, if present, must be callabe.");
			}

			Py_INCREF(create_value);
		}

		recordlist->fields_ptr[field_spec_idx].value_builder = create_value;
		recordlist->fields_ptr[field_spec_idx].column = index;
		recordlist->fields_ptr[field_spec_idx].type = type;

		strncpy(recordlist->fields_ptr[field_spec_idx].name, name, MAX_FIELDNAME_SIZE);

		field_spec_idx++;
	}

	// Sort the fields by column, so the parsing is easier and linear.
	qsort(recordlist->fields_ptr, field_spec_idx, sizeof(struct field_spec), compare_field_spec_by_column);

	// Fill structures for fast access to attributes.
	fill_type_indexes(recordlist->fields_ptr, field_spec_idx);

	// Call the C function for reading.
	recordlist->lst = read_table(fpath, recordlist->fields_ptr, field_spec_idx, recordlist->separator);

	if (recordlist->lst == NULL) {
		free(recordlist->fields_ptr);
		recordlist->fields_ptr = NULL;
		return PyErr_Format(PyExc_IOError, "Could not read file %s", fpath);
	}

	recordlist->lst->num_fields = field_spec_idx;
	recordlist->lst->fields = recordlist->fields_ptr;

	// Configure the primary timestamp for the fallback values
	configure_primary_tstamp(recordlist->lst);

	Py_RETURN_NONE;
}

void debug_print(const char *msg);

void debug_printf(const char *fmt, ...);

static PyObject* cparser_r_getattr(PyObject* self, const char* attrname);

static PyObject* cparser_r_primary_tstamp(PyObject* self)
{
	cparser_record* record = (cparser_record*) self;

	return cparser_r_getattr(self, record->lst->fields[record->lst->primary_tstamp_field].name);
}

static short can_fall_back_to_primary_tstamp(PyObject* self, const char* attrname)
{
	cparser_record* record = (cparser_record*) self;
	struct field_spec* primary_tstamp;

	if (record->lst->primary_tstamp_field == -1)
		return 0;

	primary_tstamp = &record->lst->fields[record->lst->primary_tstamp_field];

	return strcmp(primary_tstamp->name, attrname) != 0;
}

static PyObject* cparser_r_getattr(PyObject* self, const char* attrname)
{
	cparser_record* record = (cparser_record*) self;
	struct field_spec* field = NULL;
	struct record_list* lst = record->lst;
	PyObject* retval = NULL;
	uint64_t tstamp_value;
	uint8_t* src;
	size_t size;

	assert(NULL != self);
	assert(NULL != attrname);

	field = recordlist_get_fieldspec(record->lst, attrname);

	if (field == NULL) {
		return PyErr_Format(PyExc_AttributeError, "Atrribute %s not found", attrname);
	}

	//debug_printf("%s: attrname=%s\n","cparser_r_getattr",attrname);

	// AAA
	src = recordlist_get_record_field(lst, field, record->index);

	if (field->type->type == TSTAMP) {
		tstamp_value = *((uint64_t*) src);

		if (tstamp_value == -1 && can_fall_back_to_primary_tstamp(self, attrname)) // Fallback to the primary tstamp of the field in case of wrong timestamp
			retval = cparser_r_primary_tstamp(self);
		else
			retval = cparser_pandas_tstamp(tstamp_value, record->tz);
	} else {
		size = recordlist_get_record_field_size(lst, field, record->index);
		//debug_printf("size=%zu\n", size);
		retval = field->type->to_python(src, size);
	}

	// If retval is NULL because of errors in the conversion, Python will deal with it.
	return retval;
}

static
PyObject* cparser_r_str(cparser_record* self)
{
	assert(NULL != self);
	return PyString_FromFormat("cparser_record(index=%zu,parent=%p,tz=%p)", self->index, self->lst, self->tz);
}

static PyTypeObject cparser_recordType = {
	PyObject_HEAD_INIT(NULL)
	0,                         /*ob_size*/
	"cparser.Record",          /*tp_name*/
	sizeof(cparser_record),    /*tp_basicsize*/
	0,                         /*tp_itemsize*/
	(destructor) cparser_r_dealloc, /*tp_dealloc*/
	0,                         /*tp_print*/
	(getattrfunc)cparser_r_getattr, /*tp_getattr*/
	0,                         /*tp_setattr*/
	0,                         /*tp_compare*/
	0,                         /*tp_repr*/
	0,                         /*tp_as_number*/
	0,                         /*tp_as_sequence*/
	0,                         /*tp_as_mapping*/
	0,                         /*tp_hash */
	0,                         /*tp_call*/
	(reprfunc)cparser_r_str,   /*tp_str*/
	0,                         /*tp_getattro*/
	0,                         /*tp_setattro*/
	0,                         /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT,        /*tp_flags*/
	"A record object"          /* tp_doc */
};

Py_ssize_t cparser_rl_size(PyObject* self)
{
	cparser_recordlist* recordlist = (cparser_recordlist*) self;
	assert(NULL != self);
	return recordlist->lst->total_records;
}

PyObject* cparser_rl_getitem(PyObject* self, Py_ssize_t i)
{
	cparser_recordlist* recordlist = (cparser_recordlist*) self;
	cparser_record* record = NULL;

	assert(NULL != self);

	if (i < 0 || i >= recordlist->lst->total_records) {
		return PyErr_Format(PyExc_IndexError, "Index %zu out of range (object length is %zu)", i, recordlist->lst->total_records);
	}

	record = PyObject_New(cparser_record, &cparser_recordType);

	if (record == NULL) {
		return NULL;
	}

	record->tz = recordlist->tz;
	record->lst = recordlist->lst;
	record->index = i;

	Py_XINCREF(record->tz);

	return (PyObject*) record;
}

PyObject* cparser_rl_get_min_tstamp(cparser_recordlist *self, void* __attribute__((__unused__)) closure)
{
	assert(NULL != self);
	return cparser_pandas_tstamp(self->lst->min_tstamp, self->tz);
}

PyObject* cparser_rl_get_max_tstamp(cparser_recordlist *self, void* __attribute__((__unused__)) closure)
{
	assert(NULL != self);
	return cparser_pandas_tstamp(self->lst->max_tstamp, self->tz);
}


PyObject* cparser_rl_get_tz(cparser_recordlist *self, void* __attribute__((__unused__)) closure)
{
	assert(NULL != self);
	Py_INCREF(self->tz);
	return self->tz;
}

static PyMethodDef cparser_rl_methods[] = {
	{ "read", (PyCFunction) cparser_rl_read, METH_VARARGS, "Reads a set of fields from a file."},
	{ "tstamp_range", (PyCFunction) cparser_iter_tstamp_range_from_rl, METH_VARARGS, "Get an iterator for the records in a certain time range."},
	{ "filter_fields", (PyCFunction) cparser_iter_filter_field_from_rl, METH_VARARGS, "Get an iterator for the records with a field."},
	{ "field_as_numpy", (PyCFunction) cparser_iter_field_as_numpy_from_rl, METH_VARARGS,
		"Create a numpy array for the given field.\n"
		":param field: string with the name of the field.\n"
		":param fill: if the field is a timestamp, this value will be used instead of an invalid timestamps (<= 0).\n"
		"             Optional argument. Defaults to 0. NOTE: the value should be a timestamp in NANOSECONDS."
	},
	{NULL}  /* Sentinel */
};

static PySequenceMethods cparser_rl_seqmethods = {
	cparser_rl_size,    /* sq_length */
	0,                  /* sq_concat */
	0,                  /* sq_repeat */
	cparser_rl_getitem, /* sq_item   */
	0,                  /* sq_ass_item */
	0,                  /* sq_contains */
	0,                  /* sq_inplace_concat */
	0                   /* sq_inplace_repeat */
};

static PyGetSetDef cparser_rl_getsetters[] = {
	{"min_timestamp", (getter)cparser_rl_get_min_tstamp, NULL,  NULL},
	{"max_timestamp", (getter)cparser_rl_get_max_tstamp, NULL,  NULL},
	{"tz", (getter)cparser_rl_get_tz, NULL,  NULL},
	{NULL}  /* Sentinel */
};

extern PyObject* cparser_rl_iter(PyObject* self); // Defined in cparser-iterator.c. Not a good idea but well...

static PyTypeObject cparser_recordlistType = {
	PyObject_HEAD_INIT(NULL)
	0,                         /*ob_size*/
	"cparser.RecordList",      /*tp_name*/
	sizeof(cparser_recordlist),/*tp_basicsize*/
	0,                         /*tp_itemsize*/
	(destructor)cparser_rl_dealloc,  /*tp_dealloc*/
	0,                         /*tp_print*/
	0,                         /*tp_getattr*/
	0,                         /*tp_setattr*/
	0,                         /*tp_compare*/
	0,                         /*tp_repr*/
	0,                         /*tp_as_number*/
	0,                         /*tp_as_sequence*/
	0,                         /*tp_as_mapping*/
	0,                         /*tp_hash */
	0,                         /*tp_call*/
	0,                         /*tp_str*/
	0,                         /*tp_getattro*/
	0,                         /*tp_setattro*/
	0,                         /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_ITER,        /*tp_flags*/
	"List of parsed records",  /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	cparser_rl_iter,           /* tp_iter */
	0,                         /* tp_iternext */
	cparser_rl_methods,        /* tp_methods */
	0,                         /* tp_members */
	cparser_rl_getsetters,     /* tp_getset */
	0,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	cparser_rl_init,           /* tp_init */
	0,                         /* tp_alloc */
	cparser_rl_new,            /* tp_new */
};

static PyMethodDef cparser_methods[] = {
	{NULL, NULL, 0, NULL}        /* Sentinel */
};

cparser_record* cparser_record_create(void)
{
	return PyObject_New(cparser_record, &cparser_recordType);
}


#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC
initcparser(void)
{
	PyObject* m;

	cparser_recordlistType.tp_as_sequence = &cparser_rl_seqmethods;

	if (PyType_Ready(&cparser_recordlistType) < 0) {
		return;
	}

	if (PyType_Ready(&cparser_recordType) < 0) {
		return;
	}

	if (cparser_pandas_init() < 0) {
		return;
	}

	m = Py_InitModule3("cparser", cparser_methods,
	                   "Cparser module for fast file reading.");

	import_array();

	init_type_interact();

	Py_INCREF(&cparser_recordlistType);
	Py_INCREF(&cparser_recordType);
	PyModule_AddObject(m, "RecordList", (PyObject *)&cparser_recordlistType);

	cparser_iter_init_types(m);
}
