C parser module (ccore)
===============

Motivation
-----
This module was first developed for the project [FERMIN](www.naudit.es/en/fermin/) aiming to improve speed and memory footprint of the reading operations. This reading operations were performed over text files with enriched records obtained from network dissector tools. Afterwards, high level python parsers process their content for multiple and diverse analysis purposes. The examples described in the following document refer to these dissector parsers as the upper objects who make use of the ccore, but we note that this module could be used for any other kind of text file.

To improve speed and memory footprint, the dissector parsers can use the C parser module to read from files. This module also provides functions that will improve performance of filtering and timeseries generation.

Usage
-----

First, the `:py:class:cparser.RecordList` object must be created, and then the `:py:func:cparser.RecordList.read` method called with the appropiate parameters.

Once all the values are read from the file and saved in the corresponding blocks after calling the read method, the records can be accessed. The `:py:class:cparser.RecordList` class exposes list methods, so it can be accessed by index or iterated over.

The elements of the list are actually an internal type that only haves the index of the record. However, when attributes of the record are accessed, the type builds the Python objects on demand and returns them to the caller, so the implementation details are transparent to the user and the record behaves like any other Python class.

```python
	import cparser

	file_path = 'file.dat'
	parser = cparser.RecordList()  # Create the parser

	# Dictionary with the parameters of the file.
	# The keys are the field names, and the values are tuples
	# with the field specification: the column index, the type and,
	# optionally, a function to convert the value from the text.
	params = {
		'srcMAC': (0, 'string'),
		'dstMAC': (1, 'string'),
		'frames': (2, 'int'),
		'bytes': (3, 'int'),
		'ts_start': (4, 'double'),
		'ts_end': (5, 'double'),
		'is_vlan': (6, 'bool', lambda word: int(word) > 0)  # Functions can be used to calculate values to store depending on the column value.
	}

	parser.read(file_path, params)
	num_iterated = 0

	for record in parser:
		print record.srcMAC, record.dstMC

	print parser[0].srcMAC

	vlan_records = parser.filter_fields(True, 'is_vlan')  # Get all records with VLAN
	frames = vlan_records.field_as_numpy('frames')  # Get a numpy array of all the frames

```

See the *test_cparser.py* test file for examples.


Class cparser.RecordList documentation
-------------------

### method:: __init__(timezone = None, separator = " ")

>  Initializes the recordlist.
>
>  :param timezone: If not none, the Pandas timestamps will be created with this timezone.
>  :param separator: String with the field separator. Defaults to ' ' (space). Note that cparser behaves like awk regarding separator: multiple separators are ignored only when they are spaces or tabs, not when they are other character. 
>
> For example, ``hello    world`` is counted as two fields even when there are multiple spaces separating them, but ```hello||world``` with | as separator is counted as three fields.

### method:: read(fpath, field_dict)

> Reads and parses the file, storing the fields in internal C structures.
>
>  :param fpath: Path to the file to parse.
>  :param field_dict: Dictionary with the fields to parse, where the keys are the field names and the values are tuples (column_index, type, [func]). Type is a string containing one of the following types:
>
>  * *int*
>  * *string*
>  * *double*
>  * *ip*. The parser reads an IP and converts it to an integer. The Python object returned is the same integer.
>  * *tstamp*. The parser reads and stores a double value. The returned Python object is a Pandas.Timestamp object.
>  * *bool*. The parser reads an integer. The Python object returned is a *bool* object (either True or False).
>
>  Func is an optional value in the tuple. If present, it should be a function (or a lambda) that receives the word in the column and returns a value of the appropiate type to be stored.

### method:: tstamp_range(start, end, fields)

> Get an iterator that will only return records where at least one of the indicated fields is in the time range start-end.
>
> :param start: pandas.Timestamp of the range start.
> :param end: pandas.Timestamp of the range start.
> :param fields: List of fields (strings) that should be compared with the start/end times.

### method:: field_as_numpy(field, default_tstamp)

> Return in a Numpy array all the values of 'field' of the records. Especially useful for quick generation of timeseries.
>
> :param string field: Name of the field to put in the array.
> :param pandas.Timestamp default_tstamp: Default timestamp if the field is a timestamp with value 0.
> :returns: Numpy array.

### method:: filter_fields(value, fields)

> Get an interator that will only show records where at least one of the fields matches the given value.
>
> :param value: Value to filter by.
> :param fields: Fields that will be checked. All of them must be of the same type.
> :type fields: List of strings
> :returns: Iterator with the corresponding filters configured.

> .. attribute:: min_timestamp
>
> 	Minimum pandas.Timestamp value seen in the parsing. Timestamps values below 0 are ignored.
>
> .. attribute:: max_timestamp
>
>	Maximum pandas.Timestamp value seen in the parsing, or 0 if no one was found. Timestamps values below 0 are ignored.
>
> .. attribute:: tz
>
>	Timezone in which the parser is configured in.

How does it work
----------------

This module, placed in *project/ccore*, reads the fields from the file and stores them in several blocks. Each block has one array for each possible field type (currently, strings, integers and doubles), where all the field values are stored.

Each field of the parser is represented by a *field_spec* structure, defined in the *cparser.h* file. This structure defines four properties: the column index, the type, the name and the type index. The type index is the property that allows direct access to any field of any given record. For example, when parsing records with four fields, two integers and two strings, the first two integers would get type indexes 0 and 1 respectively. The two strings would get also type indexes 0 and 1, because they are respectively the first and second fields of type string in a record.

Thus, to access to an integer field with type index 2 in the record number 19823, we would have the following code:

```c

	record_index = 19823
	type_index = 2

	block_index = 19832 / records_per_block
	record_in_block_index = record_index % record_per_block
	offset_in_array = fields_per_record[integers] * record_in_block_index + type_index

	value = blocks[block_index].fields[integers][offset_in_array]
```

When a specific record is requested (either by iteration or by index access), the object returned is just a struct (*cparser_record* C type in *cparser-python.c*) holding a reference to the list and the index of the record. When a specific attribute is requested, the overridden *__getattr__* function is called, which gets the corresponding value as specified above and then builds the Python object on demand. Thus, only fields that are actually used are built, saving huge amounts of memory and computing.

Filtering
---------

In order to improve performance, the cparser implements simple filters that are computed in the C module. These are filters by range (tstamp_range) and by field equality (filter_fields). Both functions return an iterator on which further filters may be applied (only one of each type) and run utility functions such as field_as_numpy to quickly get values from records.

Adding new types
----------------

Each type is described by a *rd_type* structure, that contains the following fields:

* *type*: The type identifier, a value from the *rd_types* enum.
* *storage_type*: The type of the underlying variable used to store the field. For example, a timestamp may be saved in a variable of type double. It is a value from the *rd_base_types* enum.
* *read*: A pointer to a function that receives a string (the field to parse) and a pointer to the location where the field should be saved.
* *to_python*: A pointer to a function that receives the pointer to the value and its size, and returns the corresponding Python object.
* *from_python*: A pointer to a function that receives the PyObject* and stores the correspnding C value. Needed when optional reader functions are used.
* *has_variable_size*: Decides if this field has fixed size (numeric values) or not (strings). If it has variable size, the *read* function can be NULL: the *read_table* function will instead save the address of beginning of the field in the field storage, and will also save the length of the field. See the function *python_str* in *cparser-python.c* for an example of how these fields are managed.
* *name*: The name of the type, so it can be identified from the Python code.

In order to add new types, one should add its identifier to the *rd_types* enum in *cparser.h*, create the functions to read the value from a string and to build the Python object (you can see some examples in the read_* and python_* functions in *cparser-python.c*), and finally register the type using the function *add_type*, which receives the fields of the *rd_type* structure in the same order as in the above list.

