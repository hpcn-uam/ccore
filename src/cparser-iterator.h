#ifndef CPARSER_ITERATOR__H
#define CPARSER_ITERATOR__H

#define MAX_RANGE_FIELDS 10
#define MAX_COMPARE_MEMORY 1024

#include "cparser-python.h"

/**
 * Python object for iterator objects, with support for filtering
 */
typedef struct cparser_iter_s {
	PyObject_HEAD 					/**< Python headers */
	size_t current_index;			/**< Current index of the iterator */
	cparser_recordlist* parent;		/**< Reference to the recordlist object */
	short tstamp_range_enabled;		/**< Equals 1 if the iterator is in range filtering mode. */
	uint64_t range_min_tstamp;		/**< Minimum timestamp for the range filtering. */
	uint64_t range_max_tstamp;		/**< Max timestamp for the range filtering. */
	struct field_spec* range_fields[MAX_RANGE_FIELDS + 1]; 	/**< Fields being filtered */
	struct field_spec* equality_fields[MAX_RANGE_FIELDS + 1]; /**< Fields being filtered for equality */
	uint8_t equality_object[MAX_COMPARE_MEMORY];
	size_t equality_object_len;
	short equality_filter_enabled; 	/**< Equals 1 if the iterator is in field compare mode. */
	int length;						/**< If not -1, shows the length of this iterator */
} cparser_iter;

void cparser_iter_init_types(PyObject* module);

cparser_iter* cparser_iter_create(cparser_recordlist* parent);

PyObject* cparser_rl_iter(PyObject* self);
PyObject* cparser_iter_field_as_numpy_from_rl(PyObject* self, PyObject* args);
PyObject* cparser_iter_tstamp_range_from_rl(PyObject* self, PyObject* args);
PyObject* cparser_iter_filter_field_from_rl(PyObject* self, PyObject* args);


#endif
