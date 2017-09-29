#ifndef CPARSER_PYTHON__H
#define CPARSER_PYTHON__H

#include "cparser.h"
#include "pandas_interact.h"

#include <Python.h>

struct cparser_record_s;

/**
 * Python object for the recordlist, holding additional parameters and values
 * required for the Python interaction.
 */
typedef struct {
	PyObject_HEAD 					/**< Python headers */
	struct record_list* lst;		/**< Internal CParser structure. */
	struct field_spec* fields_ptr; 	/**< Fields that were parsed */
	PyObject* tz;					/**< Timezone used to generate pandas.Timestamp objects */
	char separator[5];				/**< Character used to separate fields */
} cparser_recordlist;

/**
 * Python object that represents a record, only holding its index and a reference
 * to the recordlist in which it's contained.
 */
typedef struct cparser_record_s {
	PyObject_HEAD 				/**< Python headers */
	size_t index;				/**< Index of this record in the list */
	struct record_list* lst;	/**< Reference to the parent */
	PyObject* tz;				/**< Timezone used to generate pandas.Timestamp objects */
} cparser_record;

cparser_record* cparser_record_create(void);

#endif
