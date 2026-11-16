# Handling errors and checking assumptions

Raise exceptions for conditions a caller can reasonably be expected to handle, and choose
the most specific built-in exception type that describes the problem, such as ValueError
for an argument with the right type but an unacceptable value, or TypeError for an
argument of the wrong type. Include the offending value in the message so that the person
reading a log can see what went wrong without reproducing it. Catch only the exceptions
you know how to handle and keep the protected region small, so that an unexpected error
elsewhere is not silently swallowed. A bare except clause also catches keyboard interrupts
and system exits, which is rarely what is wanted. When an error is caught and a different
one is raised, chain them so that the original traceback is kept. Clean up resources such
as files and network connections with a context manager, which releases them even when an
exception escapes. Assertions document assumptions during development, but they can be
disabled when Python runs with optimisation enabled, so they should never be the only
check on user input. Prefer returning early from a function when a precondition fails
rather than nesting the main logic inside several conditional blocks. Logging is better
than printing for diagnostic output in long-running programs, because log records carry a
level and a timestamp and can be filtered without changing the code that produces them.
