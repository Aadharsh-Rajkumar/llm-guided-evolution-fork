# Naming things in Python source files

Names are read far more often than they are written, so a name should say what a value
means rather than how it was computed. Module names are short and lowercase. Function and
variable names use lowercase words joined by underscores, class names join capitalised
words without separators, and names meant to stay fixed for the life of a program are
written in capital letters with underscores. A single leading underscore marks a name as
internal to its module; tools and readers treat it as a hint, not as a rule the language
enforces. Avoid single-letter names except for short loop counters and the conventional
names used in mathematics, and never use the lowercase letter l, the capital letter O or
the capital letter I on their own, because in many fonts they look like the digits one and
zero. Prefer whole words over abbreviations unless the abbreviation is more familiar than
the word it stands for. When two names refer to closely related values, give them a shared
prefix so they sort together in listings and in completion menus. A boolean reads best when
its name is a statement that can be true or false, such as is_ready or has_items. Names of
functions that return a value usually describe the value, while names of functions that
change something usually start with a verb. Renaming is cheap while a file is small and
expensive once it is imported elsewhere, so it is worth settling names early.
