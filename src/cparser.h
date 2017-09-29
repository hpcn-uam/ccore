#ifndef CPARSER_H
#define CPARSER_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <Python.h>

#include "type_interact.h"

struct field_spec {
	size_t column;					/**< Index of the column where this field is placed */
	struct rd_type* type;			/**< Type structure of the field */
	size_t type_idx;
	size_t size_idx;
	PyObject* value_builder;		/**< If not NULL, points to a Python function that builds this value from the rest of the record */
	char name[MAX_FIELDNAME_SIZE]; 	/**< Name of this field */
};

struct record_block {
	uint8_t* field_storage[RD_BASE_TYPES_COUNT];
	size_t* field_sizes;
	struct record_list* list_parent;
};

struct record_list {
	struct record_block** blocks;
	size_t num_blocks;
	size_t blocks_capacity;
	size_t fields_per_type[RD_BASE_TYPES_COUNT];
	size_t size_fields_per_record;
	size_t records_per_block;
	size_t total_records;
	char* mmapped_file;
	int fd;
	FILE* finput;
	size_t file_size;
	struct field_spec* fields;
	size_t num_fields;
	uint64_t min_tstamp, max_tstamp;
	int primary_tstamp_field; 			/**< Marks the primary timestamp field, used as a fallback value */
};

/**
 * Creates a record_list object, allocating an initial number of blocks.
 * @param  fields     List of fields to be parsed.
 * @param  num_fields Length of the fields list.
 * @return            Created object or NULL in case of error.
 */
struct record_list* recordlist_new(struct field_spec* fields, size_t num_fields);

/**
 * Deletes the given block and frees the associated memory.
 * @param block Block pointer.
 */
void recordlist_deleteblock(struct record_block* block);

/**
 * Creates a new list adding it to the record list.
 * @param  lst Recordlist.
 * @return     0 if success, -1 if error.
 */
short recordlist_newblock(struct record_list* lst);

/**
 * Reads a given file in table format, storing it in a record_list object.
 * @param  fpath      File path.
 * @param  fields     List of fields to be parsed.
 * @param  num_fields Number of fields.
 * @param  separator  Characters used for word separation.
 * @return            New record list or NULL in case of error.
 */
struct record_list* read_table(const char* fpath, struct field_spec* fields, size_t num_fields, const char* separator);

/**
 * Given a list of fields, fills the corresponding indexes of each type and field
 * for fast access to attributes.
 * @param fields Field list.
 * @param count  Number of fields.
 */
void fill_type_indexes(struct field_spec* fields, size_t count);

/**
 * Explore the list of fields and configure an appropriate one for the primary timestamp.
 * That timestamp field will be used as a fallback value in case of erroneous timestamps.
 * @param recordlist Record list.
 */
void configure_primary_tstamp(struct record_list* recordlist);

/**
 * Frees the record list and all resources associated.
 * @param lst List of records.
 */
void recordlist_delete(struct record_list* lst);

/**
 * Gets the field information for a given name.
 * @param  lst        Recordlist.
 * @param  field_name Name of the field.
 * @return            Field data or NULL if not found.
 */
struct field_spec* recordlist_get_fieldspec(struct record_list* lst, const char* field_name);

/**
 * Get the pointer to a given field of the record marked by index.
 * @param lst   Recordlist.
 * @param field Field structure.
 * @param index Index of the record.
 * @return 		Pointer to the field that should be converted to the correct type.
 */
void* recordlist_get_record_field(struct record_list* lst, struct field_spec* field, size_t index);

/**
 * Get the size of a given field. If it's not variable size, returns the size of the base type.
 *
 * @param  lst   Recordlist.
 * @param  field Field structure.
 * @param  index Index of the record.
 * @return       Field size.
 */
size_t recordlist_get_record_field_size(struct record_list* lst, struct field_spec* field, size_t index);

/**
 * Set the size of a given field. If it's not variable size, does nothing
 *
 * @param  lst   Recordlist.
 * @param  field Field structure.
 * @param  index Index of the record.
 * @param  size  Field size.
 */
void recordlist_set_record_field_size(struct record_list* lst, struct field_spec* field, size_t index, size_t size);

/**
 * Returns whether the given record is in the [min, max] (inclusive) range for any of the
 * given fields.
 *
 * @param  lst    Recordlist
 * @param  index  Record index.
 * @param  fields List of fields for which to compare.
 * @param  min    Min value.
 * @param  max    Max value.
 * @return        1 if in range, 0 if not.
 */
short recordlist_record_in_range(struct record_list* lst, size_t index, struct field_spec** fields, uint64_t min, uint64_t max);

/**
 * Returns whether the given record has any field of the fields list equal to the passed value.
 *
 * @param  lst        Recordlist
 * @param  index      Record index to compare.
 * @param  fields     Fields to compare.
 * @param  object     Object that has to match any of the given fields.
 * @param  object_len Length, in bytes, of the object to compare.
 * @return            1 if any field is equal to the wanted object, 0 if not.
 */
short recordlist_record_equal(struct record_list* lst, size_t index, struct field_spec** fields, uint8_t* object, size_t object_len);

#endif
