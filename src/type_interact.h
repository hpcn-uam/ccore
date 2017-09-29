/**
 * Header for functions interacting with the type system
 */

#ifndef TYPE_INTERACT
#define TYPE_INTERACT

#define SEC_TO_NSEC 1000000000
#define TS_GET_SECS(x)  (x / SEC_TO_NSEC)
#define TS_GET_NSECS(x) (x % SEC_TO_NSEC)

#include <stdlib.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#define MAX_FIELDNAME_SIZE 128
#define MAX_TYPES 128
#define MAX_TYPE_NAME 40

#define TS_MIN 0
#define TS_MAX 0xFFFFFFFFFFFFFFFF

// These types represent the storage class of a given type.
typedef enum {
	T_STRING = 0, T_DOUBLE,	// Basic types
	T_INT, T_SHORT, T_LONG,	// Numeric types (for size optimization)
	T_BYTE,
	RD_BASE_TYPES_COUNT		// A hack to count how many base types are there.
} rd_base_types;

// Each of these types is the "name" of the type that will be used.
typedef enum {
	STRING, DOUBLE,			// Basic types
	INT, BYTE, LONG, SHORT,	// Numeric types with different sizes
	IP, 					// Parses an IP, saves an integer
	TSTAMP,					// Timestamp, saves a long and returns a pandas.Timestamp object
	BOOL,					// Saves a byte, returns a corresponding Python bool object
	RD_TYPES_COUNT 			// Hack to count the number of types.
} rd_types;

/* Just an array for easy access and change of the size of the types.
    Must be in the same order than the values of rd_base_types
 */
extern size_t rd_base_types_sizes[]; // defined in type_interact.c

struct record_list;

/**
 * Macro to add a type to the type list easily.
 * @param  tp            Type ID (enum rd_types)
 * @param  storage       Storage type (enum rd_base_types);
 * @param  read_f        Function to read the value from a string (type_read_str).
 * @param  to_python_f   Function to convert the value to a python string (type_to_pyobject).
 * @param  variable_size 1 if the type has variable size.
 * @param  type_name     String with the type name
 * @param  from_python_f Function to get the type from a python object (type_from_pyobject);
 */
#define add_type(tp, storage, read_f, to_python_f, variable_size, type_name, from_python_f) do { \
	types[tp].type = tp; \
	types[tp].storage_type = storage; \
	types[tp].read = read_f; \
	types[tp].to_python = to_python_f; \
	types[tp].has_variable_size = variable_size; \
	types[tp].from_python = from_python_f; \
	strncpy(types[tp].name, type_name, MAX_TYPE_NAME); \
} while(0)

/** Read a type from a string (char*) and save it in the given location (void*) */
typedef void (*type_read_str)(char*, void*);

/** Create a python object from the value (void*) with a known size (size_t). */
typedef void* (*type_to_pyobject)(void*, size_t);

/** Get the C object (second argument) from the first PyObject* (first void*), and return its size. */
typedef size_t (*type_from_pyobject)(void*, void*);

struct rd_type {
	rd_types type;
	rd_base_types storage_type;
	type_read_str read;
	type_to_pyobject to_python;
	type_from_pyobject from_python;
	short has_variable_size; // For strings.
	char name[MAX_TYPE_NAME];
};

/**
 * List of available types in the system.
 *
 * Defined in type_interact.c
 */
extern struct rd_type types[RD_TYPES_COUNT];

/**
 * Find a type by name.
 * @param type_name Type name.
 * @return 			Type structure or NULL if it wasn't found.
 */
struct rd_type* find_type(const char* type_name);

/** Return a string with all the type names available in the system */
const char* type_list_string(void);

/**
 * Initialize the type structures, adding the corresponding fields.
 */
void init_type_interact(void);

/**
 * Convert a base type to the corresponding numpy type.
 * @param  type base type.
 * @return      Corresponding numpy type.
 */
enum NPY_TYPES storage_type_to_numpy(rd_base_types type);

size_t rdtype_storage_size(const struct rd_type* tp);

/**
 * Find the minimum of two timestamps in nanoseconds.
 * @param ts1 timestamp
 * @param ts2 timestamp
 * @return the corresponding timestamp.
 */
#define tsmin(ts1, ts2) ((ts1) < (ts2)? (ts1) : (ts2))

/**
 * Find the maximum of two timestamps in nanoseconds.
 * @param ts1 timestamp
 * @param ts2 timestamp
 * @return the corresponding timestamp.
 */
#define tsmax(ts1, ts2) ((ts1) < (ts2)? (ts2) : (ts1))

#endif
