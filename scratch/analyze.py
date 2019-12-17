
def pretty_print_list(list_of_strings):
    to_print = ""
    for item in list_of_strings:
        to_print += item + ", "

    return to_print[:-2]

f = open("normalization_methods.txt", "r")

normalization_methods = {}
line_count = 0
method_name = ""
for line in f:
    if line_count % 2 == 0:
        method_name = line.strip()
    else:
        if '?' in line:
            normalization_methods[method_name] = -1
        else:
            normalization_methods[method_name] = int(line)
    line_count += 1

f.close()
print(f"Number of methods: {len(normalization_methods.items())}")
popular_methods = [name for name, count in normalization_methods.items() if count > 15 or count < 0 ]
print(f"Number of popular methods: {len(popular_methods)}")
print(pretty_print_list(popular_methods))
