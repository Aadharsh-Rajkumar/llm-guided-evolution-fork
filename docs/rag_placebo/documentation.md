# Writing documentation strings

A documentation string is the first statement in a module, class or function, and tools
display it when someone asks for help on that object. The first line is a short summary
written as a command or a description, ending with a period, and fitting on one line. If
more is needed, leave a blank line and then explain what the object is for, what its
arguments mean, what it returns and which errors it can raise. Describe behaviour that a
caller relies on rather than the steps of the implementation, because the steps change
more often than the promise. Keep examples small and make sure they still run; an example
that has drifted from the code is worse than none. Public functions deserve a docstring;
short private helpers often do not, provided their names are clear. Comments serve a
different purpose: they explain why a particular choice was made where the reason is not
obvious from the code itself, for example a workaround for a known problem in a library or
an ordering that matters for performance. A comment that repeats what the next line
already says should be deleted. When code changes, update the comments and docstrings in
the same edit, since stale explanations mislead readers who trust them. Full sentences with
normal capitalisation are easier to read than fragments, and a short paragraph is usually
easier to follow than a long list of terse notes.
