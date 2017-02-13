# CS513_Assignment_1
CS 513, Introductory CUDA Program

INPUT: An integer n and an array A= (a_1,...,a_n) of floating point numbers . 

OUTPUT: An array of legth n:    (a_1, 2*a_1+ a_2,  3*a_1+2*a_2+a_3,  4*a_1+ 3*a_2+ 2*a_3 + a_4, ...)
That is, the ith member of your array must be i*a_1 + (i-1)*a_2 + ... + 2*a_{i-1} + a_i.

Design a parallel algorithm that runs in TIME(log n) and implement it on the CUDA platform.

Example input: (2,0,7)
Example output:  (2, 4, 13)
