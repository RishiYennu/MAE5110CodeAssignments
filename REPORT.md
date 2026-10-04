# Sketches of walker at key moments
![](output/assignment_2/sketch1.png)
![](output/assignment_2/sketch2.png)

# Ankle-Controller RoA
![Region of Attraction](output/assignment_2/roa.png)

# Poincare Section Choice

The Poincare section chosen was the moment the moving leg crosses over the standing leg or theta = 0 as this position is going to be the same for every step and isn't affected by the alpha or ankle_torque. Additionally, this allows us to map the value of the angular velocity at this position as we can see how
the different control inputs change the speed for every step every time it reaches this position.

# Grid Reslution

We verified the grid resolution by checking how closely the pre-calculated grid matches a continuous simular and we found the grid resolution we chose, which was 180 state points and 20 alpha points has a 92.5% accuracy, which is a better resolution than a smaller grid because 90 state points only gave a 62.5% agreement and 360 gave 90%.

# Plot Trajectory

![Region of Attraction](output/assignment_2/trajectory.png)

# Steps till Standstill

![Steps till Standstill](output/assignment_2/steps.png)

