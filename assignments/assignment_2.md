# Assignment 2

**PR opening deadline:** Thursday, September 17th, 11:59PM (midnight)  
**First Code Review deadline:** Sunday, September 20th, 11:59PM (midnight)  
**Report Submission deadline:** Wednesday, September 23rd, 11:59 (midnight)

You should open a PR to your own `main` branch with a clear title to make it easy to find. Your peer code-reviewing assignment will be posted on Canvas: go find the and give the code review by the code review deadline. You are encouraged to take the initiative and communicate actively with your reviewer, and your reviewee; e.g. if you are ready for a review before the deadline, let your reviewer know.

**Goals:**
- Implement both continuous and discrete (event-based) controllers
- Build familiarity with Poincaré sections
- Build familiarity with gridding methods
- Interact with colleagues via code-reviews professionally

## Coding Practices
### Clean Code of the week: functions and encapsulation

It's often convenient (especially for these assignments) to have all your work in a single file, so you have an immediate overview of what's happening and can easily move around and change things.
This can, however, quickly get out of hand; as it does, you should start to encapsulate things into functions or even separate modules (like the integrators and models).
What should these functions look like?

- **Do one thing, do it only, and do it well.** A function name describes what it does; it should do that and only that.
  - _bad_: 
  ```python
  def dynamics(state, params):
    [...]
    return state_derivative, energies
  ```
  - _good_:
  ```python
  def evaluate_dynamics(state, params):
    [...]
    return state_derivative
  ```

- **Don't repeat yourself** if you have copy/pasted the same code twice, you should probably encapsulate it. Repeated code is very error prone, since it's easy to forget to apply changes everywhere.

- **One level of abstraction**
    If you're reading a paper that keeps alternating between high-level concepts and nitty-gritty details, it creates a heavier cognitive load, as you keep multiple levels of abstraction in your head. It's the same with code. Use functions to encapsulate details so that you can read the script at one level of abstraction: the level you're currently developing in.
    - _good while developing the controller_:
    ```python
    for loop:
        control_input = Kp * (state[0] - desired_state[0]) + Kd * (state[1] - desired_state[1])
        params["ankle_torque"] = np.clip(control_input, min_torque, max_torque)
        next_state = integrate(model, params)
    ```
    - _good once finished with controller development_:
    ```python
    for loop:
        params["ankle_torque"] = compute_ankle_torque(state, params)
        next_state = integrate(model, params)
    ```

- **Make intent explicit** Just as function names should start with a verb describing what will happen, their signature should make their intent explicit as well.
  - _bad_:
  ```python
    def compute_ankle_torque(state, params):
        params["ankle_torque"] = Kp * (state[0] - desired_state[0]) + Kd * (state[1] - desired_state[1])
        return
    ```
  - _good_:
  ```python
    def compute_ankle_torque(state, params):
        control_input = Kp * (state[0] - desired_state[0]) + Kd * (state[1] - desired_state[1])
        return np.clip(control_input, min_torque, max_torque)
    ```
  In the bad example, reading the function call keeps the change to the `params` implicit, and a reader wouldn't know unless they went to look at the function implementation. The good example forces the function call to make the effect explicit.

## Model: the Inverted Pendulum Walker

This week's model is a slight variation of the rimless wheel that introduces two control inputs: $u = \left[u_1, u_2 \right] = \left[\tau, \alpha \right]$, where $\tau$ is a small torque that can be applied to the pivot joint of the pendulum (the "ankle"), and $\alpha$ is half the angle between stance and swing legs, just like in the rimless wheel.
Your task will be to solve for a policy that, for a slope incline $\gamma = 0.06$ rad, will bring the walker to a standing equilibrium in as few steps as possible.

We're still going to model stance as a simple inverted pendulum, but instead of defining the angle $\alpha$ between the stance and swing leg by the number of spokes in a rimless wheel, we will treat that angle as a control input that can be freely chosen once per stance phase.
In other words, you'll implement a continuous-time balancing controller with $u_1 = \tau$ in the integration loop of the pendulum dynamics, and separately a discrete-time step controller for the discrete-time Poincaré map dynamics.

### Model Implementation

Start by sketching out the inverted pendulum, and try sketching a few possible snapshots of the walker at key moments, such as mid-stance, a couple of impacts with different choices of landing angle-of-attack $\alpha$, and a failure mode.
Next, sketch the state space plots of the inverted pendulum, and for each of your snapshots, identify where the system is. Sketch out the corresponding event guards as $\alpha$ changes.

When implementing the code, it is natural here to want to change the signature of your model's dynamics call to also include the control input, such as `dynamics(t, state, control, params)`.
Instead, we'll do something simpler and more flexible: keep `dynamics(t, state, params)`, and simply add in the variable we want to use as a control into the parameter-dictionary.
This way, you can easily change which parameters to choose as control inputs without changing your implementation.
Since we are calling the integrator in the experiment script for each timestep, the controller will live at the same level and can be modified in the script directly.

Complete the model functions in the code stub [`models/inverted_pendulum_walker.py`](../models/inverted_pendulum_walker.py), based on the rimless wheel, and add parameter indicating the ankle torque `ankle_torque`, which should be summed to the pendulum dynamics (set this as 0 by default). The rest of the continuous-time dynamics shouldn't need to change.

We will create a policy that chooses the next landing angle of attack $\alpha$ once per step, and the ankle torque $\tau$ at every simulation timestep.
In your script, you'll also define the following bounds: $\alpha \in \left[ \frac{\pi}{8}, \frac{\pi}{7} \right]$, and $\tau \in \left[-0.1mg\ell, 0.05mg \ell \right ]$, which your controller will need to respect.

### Visualizing the walker

Use [`assignment_2.py`](../assignment_2.py) as a starting point for your experiment. Run `uv run python assignment_2.py` to display an animation and save `output/assignment_2/walker.gif`. The placeholder dynamics keep the pose fixed until you implement the continuous dynamics and impact functions.

The visualizer expects `[theta, angular_velocity]`, with `theta` measured clockwise from upward vertical and angles in radians. Match the parameter names in the starter script; you shouldn't need to, but you can always look under the hood in [`visualize`](../models/inverted_pendulum_walker.py).

### Stabilize the upright equilibrium with Feedback Linearization

As your first step, implement feedback linearization to cancel out the pendulum dynamics, and then add a term to ensure the pendulum's upright equilibrium is stabilized.
You are free to pick your strategy for this; inverting gravity and adding some damping, however, is a simple approach.
Use a small grid-search (think of the obvious bounds to keep this focused) to then determine the region of attraction (RoA) for your controller.

Any time your system's state enters this RoA, you're done!

Implement an event guard to detect reaching the RoA; outside, leave the ankle controller off.
### Choosing your Poincaré Section

In the rimless wheel, it was convenient to choose the touch-down event as the Poincaré section, since geometry dictated a constant touch-down angle $\theta_{\text{TD}}(\gamma, \alpha)$, and so we only had to sweep the angular velocities $\dot{\theta}$.
You should have realized from your sketches, now that we are allowing $\alpha$ to change, the touch-down guard changes with the control input.
Let's make our lives easier again by picking a new, simpler Poincaré section.
Use your sketch of the state-space to pick a new Poincaré section that:
- is transverse to the (relevant) flow (orbits).
- allows you to use $\dot{\theta}_k$ as the only state for your Poincaré return map. In other words, $\theta$ should be a constant.
For the next part, we will deal with discrete-time, step-to-step dynamics of the walker, with a single state $\dot{\theta}_k$ and single control input $u = u_2 = \alpha$.

### Control as a lookup table
Next, you want to create a look-up table so that, for any initial condition, you can find the sequence of footsteps that brings you into the RoA of your standing controller.
You'll create a 2D table for this, with the state on one axis and the control input on the other. In reinforcement learning parlance, this table is represenating the _state-action space_ (state-'control-input' space) of the step-to-step dynamics of the walker.

For the states, sweep initial angular velocities between zero and $\dot{\theta} = \sqrt{2 \frac{g}{\ell}}$, which is based on reaching a [Froude number of 2](https://www.sciencedirect.com/science/article/pii/S0966636204000268?casa_token=meVbupO-zQ8AAAAA:XxAS1QOrU4YuBgSAPBe1Fm3FsncZetZ8anQn7YifLLJiyHv-jTBFn3jlWgm2LC0xoMnizUqCFA).
For control, sweep the range of permissible inputs.
For each of these, simulate forward to find the next iterate of the state $\dot{\theta}_{k+1}$.

Some of these should already land in the RoA of your stabilizing ankle-controller; any of these bring you to upright standing in a single step!

For the rest, it is highly unlikely that $\dot{\theta}_{k+1}$ lands exactly on one of your grid-points. As long as your grid resolution is fine enough, you can use the closest state to look up which action to use next. How will you test if your grid resolution is fine enough? Decide on a criteria and use it to find the coarsest grid (this will make things faster).

Starting from states that have a controller that stabilizes the system in a single step, back out which states can come to a standstill in two steps, in three steps, etc.

Visualize this with a plot.

### Deliverables

Open a pull request (PR) from your branch into the `main` branch of your own fork.

Also, create a markdown file reporting, showing:
- Your sketches
- a visualization of the region of attraction for your ankle-controller
- Explain your choice of Poincaré section
- Explain how you verified your grid resolution, and show numbers to support your final decision (you should show numbers that show that a slightly lower resolution would _not_ be good enough).
- Plot the trajectory for an initial condition that requires at least 3 steps. For this initial condition, find and plot also the maximum number of steps the walker can continue walking before reaching the RoA.
- a visualization of how many steps it takes to reach standstill for a given initial condition.

Compile this markdown file into a PDF, using `pandoc` or similar, and submit the PDF on Canvas.

**Note:** you should absolutely avoid committing binary files (non-text files) such as images in git. Instead, you should commit code needed to generate any artifacts you use, and compile those locally.
