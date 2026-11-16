# Layout and whitespace

Consistent layout lets a reader see structure at a glance. Indent with four spaces per
level and do not mix tabs with spaces in the same file. Keep lines reasonably short; many
projects stop at seventy-nine or ninety-nine characters so that two files fit side by side
on one screen. When a long expression has to wrap, break it inside parentheses rather than
with a trailing backslash, and line the continuation up with the opening bracket or indent
it one extra level so it is not mistaken for the body of a block. Surround top-level
definitions with two blank lines and definitions inside a class with one. Use blank lines
inside a function sparingly, to separate steps that a reader would describe with different
sentences. Put a single space on each side of assignment and comparison operators, and
around the lowest-priority operators in a longer arithmetic expression, but not inside
brackets or immediately before a comma, colon or closing bracket. Keyword arguments and
default values are written without spaces around the equals sign. One statement per line
is easier to step through in a debugger than several statements joined by semicolons.
Imports go at the top of the file, one module per line, grouped from the standard library
to third-party packages to local modules, with a blank line between the groups. Trailing
whitespace at the end of a line adds noise to version-control differences and is best
removed automatically by the editor.
