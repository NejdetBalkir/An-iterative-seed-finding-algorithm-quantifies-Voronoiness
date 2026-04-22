Here, we detail the analytical derivation of deviation from Voronoiness in a symmetrically perturbed honeycomb tessellation. The distance between the neighboring honeycomb centers is set to $1$. We start by moving the walls of the central cell by $\Delta a$ as shown in the Figure. As a result, the reflected points in its six neighbors are pushed outwards. As our algorithm runs, it pushes the seeds into those neighboring cells as well. The reflection of the seeds from the first nearest neighbor over to the third nearest neighbor moves closer to the center while pulling the seeds in the third nearest neighbor. Reflections of this movement to the second-nearest neighbor also pull them towards the center. All these movements are displayed in the Figure. The displacements at the $i^\mathrm{th}$ neighbor are denoted by $d_i$. Displacement at the central cell is zero due to symmetry. Displacements at the $4^\mathrm{th}$ neighbor are assumed to be zero. This is an approximation rooted in the fact that the movements of reflections are averaged out, leading to exponential decay of perturbations. Reflected displacements are shown in a separate panel. Directions of the arrows are accurate, but the lengths are not up to scale. In particular, $\Delta a > d_1$ as shown below, but it doesn't appear so in the figure. In other words, the displacements in seeds are plotted with exaggeration with respect to the wall displacement.

Once the algorithm reaches self consistency, the sum of the reflected displacements (shown in the right panel) should be equal to the seed displacements (shown in the left panel). This gives us the following equations.

$$
d_1 = \frac{1}{6} \left(2d_1 + d_3 + 2 \Delta a \right)
$$

$$
d_2 = \frac{\sqrt{3}}{6} d_3
$$

$$
d_3 = \frac{1}{6} \left(d_1 + \sqrt{3} d_2 \right)
$$

Solving simultaneously we get the following.

$$
d_1 = \frac{11}{21} \Delta a
$$

$$
d_2 = \frac{\sqrt{3}}{63} \Delta a
$$

$$
d_3 = \frac{2}{21} \Delta a
$$

The deviation from Voronoiness for the central cell is given by the following formula.

$$
\Delta V^0 = \frac{2\Delta a - d_1}{1 + d_1} = \frac{31}{21} \Delta a + \mathcal{O}(\Delta a^2)
$$

Similarly we can ignore the deviation from unity in the denominator to write $\Delta V^i$ up to $\mathcal{O}(\Delta a^2)$ for $i = 1, 2,$ and $3$.

$$
\Delta V^1 = \frac{1}{6} \left((2 - d_1) + 2 \sqrt{d_1^2 + d_2^2} + (d_1 - d_3)) \right) + \mathcal{O}(\Delta a^2)
$$

$$
\Delta V^2 = \frac{1}{6} \left( 2 \sqrt{d_1^2 + d_2^2} + 2 \sqrt{(d_2 - \sqrt{3} d_3 /2)^2 + d_3^2/4} \right) + \mathcal{O}(\Delta a^2)
$$

$$
\Delta V^3 = \frac{1}{6} \left(  (d_1 - d_3) + 2 \sqrt{(d_3 - \sqrt{3} d_2 / 2)^2 + d_2^2/4} \right) + \mathcal{O}(\Delta a^2)
$$

We also performed a numerical calculation using a honeycomb tesselation with 1115 cells tiling a square-like space with side length 30. The algorithm was run for 100 steps with $\Delta a = 0.001$. The table below summarizes the findings.

Analytical Numerical

$d_1$ 0.5238 0.5265
$d_2$ 0.0275 0.0259
$d_3$ 0.0952 0.1062
$\Delta V^0$ 1.4762 1.4749
$\Delta V^1$ 0.4923 0.4909
$\Delta V^2$ 0.1991 0.2185
$\Delta V^3$ 0.0957 0.1406