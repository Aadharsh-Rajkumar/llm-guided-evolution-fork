# Reducing gate count without breaking a Qiskit circuit

Plain gate count = number of instructions in `qc.data` (barriers excluded). Ways to
lower it while keeping the circuit valid:
- Use a linear chain instead of a ring (saves one two-qubit gate per layer).
- An RZ applied directly to |0> (or right before a Z-basis readout) only adds a phase
  and does not change measured probabilities - it can be removed.
- Fewer layers (`N_LAYERS`) remove a full rotation + entangling block each.
- Merge rotations: RY(a) followed by RY(b) on the same qubit equals RY(a + b).
- Replace an RY+RZ pair with one parameterized gate only if the function still
  returns the matching offset/parameter count.
If you remove gates that consumed trainable parameters, also shrink the
`ParameterVector` size in the circuit-assembly function or keep offsets consistent,
so every weight in the returned list is actually used by the circuit.
