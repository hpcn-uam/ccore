#include "cparser-iterator.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>

void debug_print(char *msg);

void cparser_iter_dealloc(cparser_iter* self)
{
	if (NULL == self) {
		return ;
	}

	Py_CLEAR(self->parent);
	self->ob_type->tp_free((PyObject*) self);
}

PyObject* cparser_rl_iter(PyObject* self)
{
	assert(NULL != self);

	cparser_recordlist* recordlist;
	cparser_iter* iter;

	recordlist = (cparser_recordlist*) self;
	//debug_print("cparser_rl_iter 1");
	iter = cparser_iter_create(recordlist);
	iter->current_index = 0;

	//debug_print("cparser_rl_iter 2");
	return (PyObject*) iter;
}

PyObject* cparser_iter_iter(PyObject* self)
{
	assert(NULL != self);

	cparser_iter* iter = (cparser_iter*) self;

	//debug_print("cparser_iter_iter 1");

	Py_INCREF(self);
	iter->current_index = 0;

	//debug_print("cparser_iter_iter 2");

	return self;
}

static short cparser_iter_idx_passes_filter(cparser_iter* iter, size_t idx)
{
	assert(NULL != iter);

	if (iter->tstamp_range_enabled) {
		short passes_range = recordlist_record_in_range(iter->parent->lst, idx,
		                     iter->range_fields, iter->range_min_tstamp,
		                     iter->range_max_tstamp);

		if (!passes_range) {
			return 0;
		}
	}

	if (iter->equality_filter_enabled) {
		short passes_equal = recordlist_record_equal(iter->parent->lst, idx,
		                     iter->equality_fields, iter->equality_object, iter->equality_object_len);

		if (!passes_equal) {
			return 0;
		}
	}

	return 1;
}


PyObject* cparser_iter_iternext(PyObject* self)
{
	assert(NULL != self);

	cparser_iter* iter = (cparser_iter*) self;
	cparser_recordlist * recordlist;
	cparser_record* record = NULL;
	size_t idx;

	recordlist = iter->parent;
	idx = iter->current_index;

	// If in range filtering mode, discard records that do not pass the filter.
	while (idx < recordlist->lst->total_records && !cparser_iter_idx_passes_filter(iter, idx)) {
		idx++;
	}

	if (idx >= recordlist->lst->total_records) {
		PyErr_SetNone(PyExc_StopIteration);
		return NULL;
	}

	record = cparser_record_create();

	if (record == NULL) {
		return NULL;
	}

	record->lst = recordlist->lst;
	record->index = idx;
	record->tz = recordlist->tz;

	Py_XINCREF(record->tz);

	iter->current_index = idx + 1;

	return (PyObject*) record;
}

PyObject* cparser_iter_field_as_numpy(PyObject* self, PyObject* args)
{
	assert(NULL != self);

	npy_intp arr_dims[1];
	PyArrayObject* array;
	const char* field_name;
	cparser_iter* iter = (cparser_iter*) self;
	cparser_recordlist * recordlist = iter->parent;
	void* src, *dst;
	uint64_t tstamp;
	int64_t* tstamp_dst;
	size_t field_size;
	uint64_t default_tstamp = 0;
	npy_intp dst_i = 0;
	npy_intp _dims[1];
	PyArray_Dims new_dims;

	new_dims.ptr = _dims;
	new_dims.len = 1;

	struct field_spec* field;

	if (!PyArg_ParseTuple(args, "s|L", &field_name, &default_tstamp)) {
		return NULL;
	}

	field = recordlist_get_fieldspec(recordlist->lst, field_name);

	if (field == NULL) {
		return PyErr_Format(PyExc_TypeError, "Unknown field %s", field_name);
	}

	if (field->type->has_variable_size) {
		return PyErr_Format(PyExc_TypeError, "Can't create arrays if field has variable size");
	}

	field_size = rd_base_types_sizes[field->type->storage_type];

	if (iter->length == -1) {
		arr_dims[0] = recordlist->lst->total_records;
	} else {
		arr_dims[0] = iter->length;
	}

	if (field->type->type == TSTAMP) {// Special case
		// We need to specify the units of the timestamp.
		PyObject *dtype_specifier = Py_BuildValue("s", "datetime64[ns]");
		PyArray_Descr *descr;
		PyArray_DescrConverter(dtype_specifier, &descr);

		array = (PyArrayObject*) PyArray_SimpleNewFromDescr(1, arr_dims, descr);

		if (array == NULL) {
			return NULL;
		}

		dst_i = 0;

		for (npy_intp i = 0; i < recordlist->lst->total_records; i++) {
			if (!cparser_iter_idx_passes_filter(iter, i)) {
				continue;
			}

			src = recordlist_get_record_field(recordlist->lst, field, i);
			tstamp_dst = PyArray_GETPTR1(array, dst_i);
			dst_i++;

			tstamp = *((uint64_t*) src);

			if (tstamp == 0 || tstamp == (uint64_t)-1) {
				tstamp = default_tstamp;
			}

			*tstamp_dst = tstamp;
		}
	} else {
		array = (PyArrayObject*) PyArray_SimpleNew(1, arr_dims, storage_type_to_numpy(field->type->storage_type));

		if (array == NULL)
			return NULL;

		dst_i = 0;

		for (npy_intp i = 0; i < recordlist->lst->total_records; i++) {
			if (!cparser_iter_idx_passes_filter(iter, i)) {
				continue;
			}

			src = recordlist_get_record_field(recordlist->lst, field, i);
			dst = PyArray_GETPTR1(array, dst_i);
			dst_i++;

			memcpy(dst, src, field_size);
		}
	}

	_dims[0] = dst_i;

	if (_dims[0] != arr_dims[0]) {
		PyArray_Resize(array, &new_dims, 0, NPY_ANYORDER);
	}


	return (PyObject*) array;
}

PyObject* cparser_iter_field_as_numpy_from_rl(PyObject * self, PyObject * args)
{
	assert(NULL != self);
	assert(NULL != args);

	cparser_iter* iter = cparser_iter_create((cparser_recordlist*) self);
	PyObject* retval = cparser_iter_field_as_numpy((PyObject*) iter, args);

	Py_CLEAR(iter);
	return retval;
}

static cparser_iter* cparser_iter_duplicate(cparser_iter * iter_orig)
{
	assert(NULL != iter_orig);

	cparser_iter* iter = cparser_iter_create(iter_orig->parent);

	if (iter_orig->equality_filter_enabled) {
		iter->equality_filter_enabled = 1;
		iter->equality_object_len = iter_orig->equality_object_len;
		memcpy(iter->equality_object, iter_orig->equality_object, iter->equality_object_len);
		memcpy(iter->equality_fields, iter_orig->equality_fields, (MAX_RANGE_FIELDS + 1) * sizeof(struct field_spec*));
	}

	if (iter_orig->tstamp_range_enabled) {
		iter->tstamp_range_enabled = 1;
		iter->range_max_tstamp = iter_orig->range_max_tstamp;
		iter->range_min_tstamp = iter_orig->range_min_tstamp;
		memcpy(iter->range_fields, iter_orig->range_fields, (MAX_RANGE_FIELDS + 1) * sizeof(struct field_spec*));
	}

	return iter;
}

PyObject* cparser_iter_tstamp_range(PyObject * self, PyObject * args)
{
	assert(NULL != self);

	cparser_iter* iter_orig = (cparser_iter*) self;
	cparser_iter* iter = cparser_iter_duplicate(iter_orig);
	cparser_recordlist * recordlist = iter_orig->parent;
	PyObject * fieldseq, *fieldseq_iter;
	PyObject* str_obj;
	size_t num_fields = 0;
	struct field_spec* field;
	const char* field_name;
	short error = 0;

	if (!PyArg_ParseTuple(args, "LLO", &iter->range_min_tstamp, &iter->range_max_tstamp, &fieldseq)) {
		return NULL;
	}

	fieldseq_iter = PyObject_GetIter(fieldseq);

	if (fieldseq_iter == NULL) {
		return PyErr_Format(PyExc_TypeError, "Argument should be an iterable of strings (it is not iterable)");
	}

	while ((str_obj = PyIter_Next(fieldseq_iter)) != NULL && !error && num_fields < MAX_RANGE_FIELDS) {
		if (!PyString_Check(str_obj)) {
			PyErr_Format(PyExc_TypeError, "Argument should be an iterable of strings (the elements are not strings)");
			error = 1;
		} else {
			field_name = PyString_AS_STRING(str_obj);
			field = recordlist_get_fieldspec(recordlist->lst, field_name);

			if (field == NULL) {
				PyErr_Format(PyExc_TypeError, "Unrecognized field %s", field_name);
				error = 1;
			} else if (field->type->type != TSTAMP) {
				PyErr_Format(PyExc_TypeError, "Only timestamp fields can be used");
				error = 1;
			} else {
				iter->range_fields[num_fields] = field;
				num_fields++;
			}
		}

		Py_DECREF(str_obj);
	}

	Py_DECREF(fieldseq_iter);

	if (error || PyErr_Occurred())
		return NULL;

	iter->range_fields[num_fields] = NULL;
	iter->tstamp_range_enabled = 1;

	return (PyObject*) iter;
}

PyObject* cparser_iter_tstamp_range_from_rl(PyObject * self, PyObject * args)
{
	assert(NULL != self);

	cparser_iter* iter = cparser_iter_create((cparser_recordlist*) self);

	PyObject* retval = cparser_iter_tstamp_range((PyObject*) iter, args);

	Py_CLEAR(iter);
	return retval;
}


PyObject* cparser_iter_filter_field(PyObject * self, PyObject * args)
{
	assert(NULL != self);

	cparser_iter* iter_orig = (cparser_iter*) self;
	cparser_iter* iter = cparser_iter_duplicate(iter_orig);
	cparser_recordlist * recordlist = iter_orig->parent;
	PyObject * fieldseq, *fieldseq_iter, *filter_obj;
	PyObject* str_obj;
	size_t num_fields = 0;
	struct field_spec* field;
	const char* field_name;
	struct rd_type *type = NULL;

	short error = 0;

	if (!PyArg_ParseTuple(args, "OO", &filter_obj, &fieldseq)) {
		return NULL;
	}

	fieldseq_iter = PyObject_GetIter(fieldseq);

	if (fieldseq_iter == NULL) {
		return PyErr_Format(PyExc_TypeError, "Argument should be an iterable of strings (it is not iterable)");
	}

	while ((str_obj = PyIter_Next(fieldseq_iter)) != NULL && !error && num_fields < MAX_RANGE_FIELDS) {
		if (!PyString_Check(str_obj)) {
			PyErr_Format(PyExc_TypeError, "Argument should be an iterable of strings (the elements are not strings)");
			error = 1;
		} else {
			field_name = PyString_AS_STRING(str_obj);
			field = recordlist_get_fieldspec(recordlist->lst, field_name);

			if (field == NULL) {
				PyErr_Format(PyExc_TypeError, "Unrecognized field %s", field_name);
				error = 1;
			} else if (type != NULL && field->type != type) {
				PyErr_Format(PyExc_TypeError, "All the fields must have the same type");
				error = 1;
			} else {
				iter->equality_fields[num_fields] = field;
				num_fields++;
			}

			type = field->type;
		}

		Py_DECREF(str_obj);
	}

	Py_DECREF(fieldseq_iter);

	if (error || PyErr_Occurred()) {
		return NULL;
	}

	iter->equality_fields[num_fields] = NULL;
	iter->equality_filter_enabled = 1;

	iter->equality_object_len = type->from_python(filter_obj, iter->equality_object);
	return (PyObject*) iter;
}

PyObject* cparser_iter_filter_field_from_rl(PyObject * self, PyObject * args)
{
	assert(NULL != self);

	cparser_iter* iter = cparser_iter_create((cparser_recordlist*) self);
	PyObject* retval = cparser_iter_filter_field((PyObject*) iter, args);

	Py_CLEAR(iter);
	return retval;
}


static PyMethodDef cparser_iter_methods[] = {
	{ "tstamp_range", (PyCFunction) cparser_iter_tstamp_range, METH_VARARGS, "Get an iterator for the records in a certain time range.\n Note that timestamps MUST be in nanoseconds!"},
	{ "filter_fields", (PyCFunction) cparser_iter_filter_field, METH_VARARGS, "Get an iterator for the records with some field equal to some value."},
	{ "field_as_numpy", (PyCFunction) cparser_iter_field_as_numpy, METH_VARARGS, "Create a numpy array for the given field."},
	{NULL}  /* Sentinel */
};

PyObject* cparser_iter_get_filtering_range(cparser_iter* self, void* __attribute__((__unused__)) closure)
{
	assert(NULL != self);

	return PyBool_FromLong(self->tstamp_range_enabled);
}


PyObject* cparser_iter_get_filtering_field(cparser_iter* self, void* __attribute__((__unused__)) closure)
{
	assert(NULL != self);

	return PyBool_FromLong(self->equality_filter_enabled);
}

PyObject* cparser_iter_get_min_tstamp(cparser_iter* self, void* __attribute__((__unused__)) closure)
{
	assert(NULL != self);

	return cparser_pandas_tstamp(self->parent->lst->min_tstamp, self->parent->tz);
}

PyObject* cparser_iter_get_max_tstamp(cparser_iter* self, void* __attribute__((__unused__)) closure)
{
	assert(NULL != self);

	return cparser_pandas_tstamp(self->parent->lst->max_tstamp, self->parent->tz);
}


PyObject* cparser_iter_get_tz(cparser_iter* self, void* __attribute__((__unused__)) closure)
{
	assert(NULL != self);

	Py_INCREF(self->parent->tz);
	return self->parent->tz;
}

Py_ssize_t cparser_iter_size(PyObject* self)
{
	assert(NULL != self);

	cparser_iter* iter = (cparser_iter*) self;

	if (iter->length == -1) {
		iter->length = 0;

		for (size_t i = 0; i < iter->parent->lst->total_records; i++) {
			if (cparser_iter_idx_passes_filter(iter, i)) {
				iter->length++;
			}
		}
	}

	return iter->length;
}

static PySequenceMethods cparser_iter_seqmethods = {
	cparser_iter_size, /* sq_length */
	0,                 /* sq_concat */
	0,                 /* sq_repeat */
	0,                 /* sq_item   */
	0,                 /* sq_ass_item */
	0,                 /* sq_contains */
	0,                 /* sq_inplace_concat */
	0                  /* sq_inplace_repeat */
};

static PyGetSetDef cparser_iter_getsetters[] = {
	{"is_filtering_range", (getter)cparser_iter_get_filtering_range, NULL,  NULL},
	{"is_filtering_field", (getter)cparser_iter_get_filtering_field, NULL,  NULL},
	{"min_timestamp", (getter)cparser_iter_get_min_tstamp, NULL,  NULL},
	{"max_timestamp", (getter)cparser_iter_get_max_tstamp, NULL,  NULL},
	{"tz", (getter)cparser_iter_get_tz, NULL,  NULL},
	{NULL}  /* Sentinel */
};

PyTypeObject cparser_iterType = {
	PyObject_HEAD_INIT(NULL)
	0,                                          /*ob_size*/
	"cparser.RecordListIter",                   /*tp_name*/
	sizeof(cparser_iter),                       /*tp_basicsize*/
	0,                                          /*tp_itemsize*/
	(destructor) cparser_iter_dealloc,          /*tp_dealloc*/
	0,                                          /*tp_print*/
	0,                                          /*tp_getattr*/
	0,                                          /*tp_setattr*/
	0,                                          /*tp_compare*/
	0,                                          /*tp_repr*/
	0,                                          /*tp_as_number*/
	0,                                          /*tp_as_sequence*/
	0,                                          /*tp_as_mapping*/
	0,                                          /*tp_hash */
	0,                                          /*tp_call*/
	0,                                          /*tp_str*/
	0,                                          /*tp_getattro*/
	0,                                          /*tp_setattro*/
	0,                                          /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_ITER,  /*tp_flags*/
	"List of parsed records",                   /* tp_doc */
	0,                                          /* tp_traverse */
	0,                                          /* tp_clear */
	0,                                          /* tp_richcompare */
	0,                                          /* tp_weaklistoffset */
	cparser_iter_iter,                          /* tp_iter */
	(iternextfunc) cparser_iter_iternext,       /* tp_iternext */
	cparser_iter_methods,                       /* tp_methods */
	0,                                          /* tp_members */
	cparser_iter_getsetters                     /* tp_getset */
};


cparser_iter* cparser_iter_create(cparser_recordlist* recordlist)
{
	cparser_iter* iter = PyObject_New(cparser_iter, &cparser_iterType);

	iter->parent = recordlist;
	iter->current_index = 0;
	iter->tstamp_range_enabled = 0;
	iter->equality_filter_enabled = 0;
	iter->length = -1;

	Py_XINCREF(iter->parent);

	//debug_print("cparser_iter_create");
	//debug_printf("cparser_iter_create: num_fields=%zu\n", recordlist->lst->num_fields);

	return iter;
}

void cparser_iter_init_types(PyObject* module)
{
	import_array();

	cparser_iterType.tp_as_sequence = &cparser_iter_seqmethods;

	if (PyType_Ready(&cparser_iterType) < 0) {
		return;
	}

	Py_INCREF(&cparser_iterType);

	if (module != NULL) {
		PyModule_AddObject(module, "RecordListIter", (PyObject*) &cparser_iterType);
	}
}

