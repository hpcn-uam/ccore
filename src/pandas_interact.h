#ifndef PANDAS_INTERACT_H
#define PANDAS_INTERACT_H

#include <Python.h>
#include <inttypes.h>

/**
 * Initialize the references to the Pandas module.
 * @return  0 if no error occurred, -1 if the initialization failed.
 */
short cparser_pandas_init(void);

/**
 * Construct a pandas.Timestamp object in the given timezone.
 * @param  value Time from the epoch, in seconds.
 * @param  tz    Timezone for the timestamp.
 * @return       Python object with the pandas.Timestamp.
 */
PyObject* cparser_pandas_tstamp(uint64_t value, PyObject* tz);
#endif
