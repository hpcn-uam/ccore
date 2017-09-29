#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <math.h>


#include "cparser.h"

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

size_t get_file_size(const char* file)
{
	struct stat s;

	assert(NULL != file);

	if (stat(file, &s) == -1) {
		return 0;
	}

	return s.st_size;
}

struct record_list* recordlist_new(struct field_spec* fields, size_t num_fields)
{
	struct record_list* lst;
	size_t i;
	rd_base_types storage_type;

	assert(NULL != fields);

	lst = (struct record_list*) malloc(sizeof(struct record_list));

	if (lst == NULL) {
		return NULL;
	}

	lst->records_per_block = 1000;
	lst->blocks_capacity = 32;
	lst->blocks = calloc(lst->blocks_capacity, sizeof(struct record_block**));

	if (lst->blocks == NULL) {
		free(lst);
		return NULL;
	}

	lst->num_blocks = 0;
	lst->total_records = 0;
	lst->size_fields_per_record = 0;
	lst->primary_tstamp_field = -1;

	memset(lst->fields_per_type, 0, sizeof(size_t) * RD_BASE_TYPES_COUNT);

	for (i = 0; i < num_fields; i++) {
		storage_type = fields[i].type->storage_type;
		lst->fields_per_type[storage_type]++;

		if (fields[i].type->has_variable_size) {
			lst->size_fields_per_record++;
		}
	}

	return lst;
}

static short recordlist_ensure_capacity(struct record_list* lst)
{
	size_t new_size, i;
	struct record_block** new_list;

	assert(NULL != lst);

	if (lst->num_blocks >= lst->blocks_capacity) {
		new_size = 2 * lst->blocks_capacity;

		new_list = realloc(lst->blocks, new_size * sizeof(struct record_block**));

		if (new_list == NULL) {
			return -1;
		}

		for (i = lst->blocks_capacity; i < new_size; i++) {
			new_list[i] = NULL;
		}

		lst->blocks = new_list;
		lst->blocks_capacity = new_size;
	}

	return 0;
}

void recordlist_deleteblock(struct record_block* block)
{
	size_t i;

	if (NULL == block) {
		return ;
	}

	for (i = 0; i < RD_BASE_TYPES_COUNT; i++) {
		if (block->field_storage[i] != NULL) {
			free(block->field_storage[i]);
			block->field_storage[i] = NULL;
		}
	}

	if (block->field_sizes) {
		free(block->field_sizes);
		block->field_sizes = NULL;
	}

	free(block);
}

short recordlist_newblock(struct record_list* lst)
{
	struct record_block* block = NULL;
	size_t i, alloc_size;

	assert(NULL != lst);

	block = malloc(sizeof(struct record_block));

	if (block == NULL) {
		return -1;
	}

	block->list_parent = lst;

	memset(block->field_storage, 0, RD_BASE_TYPES_COUNT * sizeof(uint8_t*));

	for (i = 0; i < RD_BASE_TYPES_COUNT; i++) {
		alloc_size = lst->fields_per_type[i] * lst->records_per_block * rd_base_types_sizes[i];

		if (alloc_size > 0) {
			block->field_storage[i] = malloc(alloc_size);

			if (block->field_storage[i] == NULL) {
				recordlist_deleteblock(block);
				return -1;
			}
		}
	}

	if (lst->size_fields_per_record > 0) {
		alloc_size = lst->size_fields_per_record * lst->records_per_block * sizeof(size_t);
		block->field_sizes = malloc(alloc_size);

		if (block->field_sizes == NULL) {
			recordlist_deleteblock(block);
			return -1;
		}
	} else {
		block->field_sizes = NULL;
	}

	if (recordlist_ensure_capacity(lst) != 0) {
		recordlist_deleteblock(block);
		return -1;
	}

	lst->blocks[lst->num_blocks] = block;
	lst->num_blocks++;

	return 0;
}

void debug_print(const char *msg)
{
	FILE *f = fopen("debug.txt", "a");
	fprintf(f, "%s\n", msg);
	fclose(f);
}

#include <stdarg.h>
void debug_printf(const char *fmt, ...)
{
	va_list ap;
	FILE *f = fopen("debug.txt", "a");
	va_start(ap, fmt);
	vfprintf(f, fmt, ap);
	va_end(ap);
	fclose(f);
}

static short is_string_blank(const char* str)
{
	// I'm Kernigan
	for (; *str != '\0'; str++)
		if (*str != '\t' && *str != ' ')
			return 0;

	return 1;
}

struct record_list* read_table(const char* fpath, struct field_spec* fields, size_t num_fields, const char* separator)
{
	char* word, *cursor;
	size_t block_idx = -1, record_idx = 0, column_idx = 0;
	struct record_list* lst;
	size_t field = 0;
	size_t file_offset = 0;
	ssize_t line_length;
	char* addr;
	char* line = NULL;
	size_t line_size;
	short ignore_empty_fields = is_string_blank(separator);

	assert(NULL != fpath);
	assert(NULL != fields);
	assert(NULL != separator);
	assert(0 != num_fields);

	lst = recordlist_new(fields, num_fields);

	if (NULL == lst) {
		return NULL;
	}

	lst->mmapped_file = MAP_FAILED;
	lst->file_size = get_file_size(fpath);
	lst->finput = fopen(fpath, "r");

	if (NULL == lst->finput) {
		perror("fopen");
		goto error;
	}

	lst->fd = fileno(lst->finput);
	lst->fields = fields;
	lst->num_fields = num_fields;
	lst->min_tstamp = TS_MAX;
	lst->max_tstamp = TS_MIN;

	if (lst->file_size != 0) {
		// Use MAP_PRIVATE is equivalent to reading the entire file in memory if we write to it
		lst->mmapped_file = mmap(NULL, lst->file_size, PROT_READ, MAP_SHARED, lst->fd, 0);

		if (lst->mmapped_file == MAP_FAILED) {
			perror("mmap");
			goto error;
		}
	}

	line_size = 200000; // This is procesaConexiones creating incredibly long lines

	if (NULL == (line = malloc(line_size))) {
		perror("line buffer malloc");
		goto error;
	}

	while ((line_length = getline(&line, &line_size, lst->finput)) != -1) {
		if (record_idx == 0) {
			block_idx++;
			recordlist_newblock(lst);
		}

		cursor = line;
		column_idx = 0;
		field = 0;

		while (field < num_fields && (word = strsep(&cursor, separator)) != NULL) {
			size_t word_len = strlen(word);

			if (word_len > 0 && word[word_len - 1] == '\n') {
				word_len--;
				word[word_len] = 0;
			}

			if (word_len == 0 && ignore_empty_fields)
				continue;

			// Check all the fields that want this column.
			while (field < num_fields && column_idx == fields[field].column) {
				struct rd_type* type = fields[field].type;
				//debug_printf("%zu, %zu\n", field, column_idx);
				uint8_t* dest = recordlist_get_record_field(lst, &fields[field], lst->total_records);

				if (fields[field].value_builder == NULL) {
					// Read the columns on non-buildable types

					if (type->has_variable_size) {
						addr = lst->mmapped_file + file_offset + (word - line);

						*((char**) dest) = addr;
						recordlist_set_record_field_size(lst, &fields[field], lst->total_records, word_len);
					} else {
						if (word_len > 0)
							type->read(word, (void*) dest);
						else
							memset((void*) dest, 0, rdtype_storage_size(type));
					}

					if (type->type == TSTAMP && word_len > 0) {
						uint64_t tstamp = *((uint64_t*) dest);

						if (tstamp != (uint64_t)-1 && tstamp != 0) {
							lst->min_tstamp = tsmin(tstamp, lst->min_tstamp);
							lst->max_tstamp = tsmax(tstamp, lst->max_tstamp);
						}
					}
				} else {
					// This field must be created by calling a Python function on the current register.
					PyObject* pystr = PyString_FromString(word);

					PyObject* args = PyTuple_Pack(1, pystr);
					PyObject* retval;

					retval = PyObject_Call(fields[field].value_builder, args, NULL);

					Py_DECREF(args);
					Py_DECREF(pystr);

					if (retval == NULL) {
						fprintf(stderr, "read_table: Fatal error: value builder for field %s (column %zu) failed at line %zu with input %s\n", fields[field].name, column_idx, record_idx + 1, word);
						fprintf(stderr, "Python traceback: \n");
						PyErr_Print();
						goto error;
					}

					type->from_python(retval, dest);

					Py_DECREF(retval);
				}

				field++;
			}

			column_idx++;
		}

		file_offset += line_length;

		if (field > 0 && column_idx > 0) {
			record_idx = (record_idx + 1) % lst->records_per_block;
			lst->total_records++;
		}
	}

	// Set sane defaults for min/max timestamps
	if (lst->min_tstamp == TS_MAX) {
		lst->min_tstamp = 0;
	}

	if (lst->max_tstamp == TS_MIN) {
		lst->max_tstamp = lst->min_tstamp;
	}

	free(line);
	return lst;

error:
	free(line);

	recordlist_delete(lst);
	return NULL;
}

void recordlist_delete(struct record_list* lst)
{
	size_t i;

	if (NULL == lst) {
		return ;
	}

	if (MAP_FAILED != lst->mmapped_file) {
		munmap(lst->mmapped_file, lst->file_size);
		lst->file_size = 0;
		lst->mmapped_file = MAP_FAILED;
	}

	if (NULL != lst->finput) {
		fclose(lst->finput);
		lst->finput = NULL;
	}

	for (i = 0; i < lst->blocks_capacity; i++) {
		recordlist_deleteblock(lst->blocks[i]);
		lst->blocks[i] = NULL;
	}

	free(lst->blocks);
	lst->blocks = NULL;
	free(lst);
}

void fill_type_indexes(struct field_spec* fields, size_t count)
{
	size_t field_nums[RD_BASE_TYPES_COUNT];
	size_t sizes = 0;
	size_t i;
	size_t storage_type;

	assert(NULL != fields);

	memset(field_nums, 0, sizeof(field_nums));

	for (i = 0; i < count; i++) {
		storage_type = fields[i].type->storage_type;

		fields[i].type_idx = field_nums[storage_type];
		fields[i].size_idx = sizes;

		field_nums[storage_type]++;

		if (fields[i].type->has_variable_size) {
			sizes++;
		}
	}
}

void configure_primary_tstamp(struct record_list* recordlist)
{
	struct field_spec* field;

	for (size_t i = 0; i < recordlist->num_fields; i++) {
		field = recordlist->fields + i;

		if (field->type->type == TSTAMP) {
			// Use the ts_start field as the timestamp fallback value when we find erroneuous timestamps
			// In case we don't find any ts_start field, use the first tstamp field we find.
			if (strcmp(field->name, "ts_start") == 0 || recordlist->primary_tstamp_field == -1)
				recordlist->primary_tstamp_field = i;
		}
	}
}

struct field_spec* recordlist_get_fieldspec(struct record_list* lst, const char* field_name)
{
	size_t i;

	assert(NULL != lst);
	assert(NULL != field_name);

	for (i = 0; i < lst->num_fields; i++) {
		if (strncmp(field_name, lst->fields[i].name, MAX_FIELDNAME_SIZE) == 0) {
			return lst->fields + i;
		}
	}

	return NULL;
}

void* recordlist_get_record_field(struct record_list* lst, struct field_spec* field, size_t index)
{
	size_t record_in_block, block_idx;
	size_t offset;
	struct rd_type* type;

	assert(NULL != lst);
	assert(NULL != field);

	//debug_print("recordlist_get_record_field");

	type = field->type;
	//debug_printf("column=%zu,rd_type=%p,type_idx=%zu,size_idx=%zu,value_builder=%p,name=%s\n", field->column, field->type, field->type_idx, field->size_idx, field->value_builder, field->name);
	//debug_printf("type=%d,storage_type=%d,read=%p,to_python=%p,from_python=%p,has_variable_size=%hd,name=%s\n", type->type, type->storage_type, type->read, type->to_python, type->from_python, type->has_variable_size, type->name);
	assert(type->type < RD_TYPES_COUNT);

	block_idx = index / lst->records_per_block;
	record_in_block = index % lst->records_per_block;

	//debug_printf("%s\n", type->name);
	assert(type->storage_type < RD_BASE_TYPES_COUNT);
	offset = lst->fields_per_type[type->storage_type] * record_in_block + field->type_idx;
	offset *= rd_base_types_sizes[type->storage_type];

	return lst->blocks[block_idx]->field_storage[type->storage_type] + offset;
}

size_t recordlist_get_record_field_size(struct record_list* lst, struct field_spec* field, size_t index)
{
	size_t record_in_block, block_idx;
	size_t offset;
	struct rd_type* type;

	assert(NULL != lst);
	assert(NULL != field);

	type = field->type;

	if (!type->has_variable_size) {
		return rd_base_types_sizes[type->storage_type];
	}

	block_idx = index / lst->records_per_block;
	record_in_block = index % lst->records_per_block;

	offset = lst->size_fields_per_record * record_in_block + field->size_idx;

	return lst->blocks[block_idx]->field_sizes[offset];
}

void recordlist_set_record_field_size(struct record_list* lst, struct field_spec* field, size_t index, size_t size)
{
	size_t record_in_block, block_idx;
	size_t offset;
	struct rd_type* type;

	assert(NULL != lst);
	assert(NULL != field);

	type = field->type;

	if (!type->has_variable_size) {
		return;
	}

	block_idx = index / lst->records_per_block;
	record_in_block = index % lst->records_per_block;

	offset = lst->size_fields_per_record * record_in_block + field->size_idx;

	lst->blocks[block_idx]->field_sizes[offset] = size;
}

short recordlist_record_in_range(struct record_list* lst, size_t index, struct field_spec** fields, uint64_t min, uint64_t max)
{
	uint64_t* value;
	struct field_spec* field;

	assert(NULL != lst);
	assert(NULL != fields);

	for (; *fields != NULL; fields++) {
		field = *fields;
		value = recordlist_get_record_field(lst, field, index);

		if (*value >= min && *value <= max) {
			return 1;
		}
	}

	return 0;
}

short recordlist_record_equal(struct record_list* lst, size_t index, struct field_spec** fields, uint8_t* object, size_t object_len)
{
	uint8_t* value;
	struct field_spec* field;
	size_t size;

	assert(NULL != lst);
	assert(NULL != fields);

	for (; *fields != NULL; fields++) {
		field = *fields;
		value = recordlist_get_record_field(lst, field, index);

		if (field->type->has_variable_size) {
			size = recordlist_get_record_field_size(lst, field, index);

			if (size != object_len) {
				continue;
			}
		}

		if (memcmp(value, object, object_len) == 0) {
			return 1;
		}
	}

	return 0;
}
