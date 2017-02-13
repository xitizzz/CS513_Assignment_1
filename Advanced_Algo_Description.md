1. Give each processor chunk of (n / N) elements and find its prefix sum locally. Make the last element as representative element. T = O(n/N)
Now we have N representative elements.
2. Find prefix sum of those N elements using parallel prefix sum algorithm. Now we have N prefix sums, one for each processors call it P.  T = O(log N)
3. Compute on each processor the final answer by adding each local element to the P of processor holding previous chunk. T = O(n/N)
