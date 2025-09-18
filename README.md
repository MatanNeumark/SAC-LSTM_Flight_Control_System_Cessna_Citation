This repository contains the source code and results of my MSc thesis at the Aerospace Engineering faculty of Delft University of Technology.
The thesis report, which contains a scientific report showcasing the main results, along with theoretical background and a preliminary analysis, can be found here:
https://repository.tudelft.nl/record/uuid:f68c711d-29f2-4c92-915b-621fa9a66026

<img width="898*0.6" height="1274*0.6" alt="image" src="https://github.com/user-attachments/assets/f92e629b-78e6-4988-9f25-7121400d2a73" />


Abstract:

Advancements in deep reinforcement learning (RL) open the door to the development
of robust flight control systems (FCS) that have the potential to improve both safety and
performance during off-nominal flight conditions. Simulation-based work on offline-RL FCS
has already demonstrated robustness to adverse weather conditions, mechanical failures, and a
wide range of operational conditions. However, it has neglected important dynamical phenomena
that limit its applicability to reality. In anticipation of a future flight testing campaign of
similar RL-based FCS, this research emulates the transition from simulation to reality by
modelling prevalent sensor and actuator dynamics, and introduces a method to incorporate
a long short-term memory (LSTM) artificial neural network (ANN) into the policy of a Soft
Actor-Critic (SAC) agent. The approach is found to largely diminish the sensitivity of the
controller to sensor noise and actuator dynamics, while increasing its robustness to delays in
comparison with the ubiquitous feedforward deep neural network (DNN) and a traditional
linear controller.


The RL-based FCS is designed for the Cessna Citation II, TU Delft's research aircraft:
<img width="4550" height="1309" alt="PH_LAB_Andre_Pronk_no_background" src="https://github.com/user-attachments/assets/6cbe4430-4239-4257-a9ca-78d918dc1375" />

Image by Andre Pronk


As shown in the control loop, the FCS controls the pitch angle of the aircraft.
<img width="1534" height="312" alt="theta_control_loop_RL_v2" src="https://github.com/user-attachments/assets/81f163a8-6bec-44a4-8c8f-70030b158de0" />

The schematic below illustrates the flow of information between the various components of the SAC-LSTM agent and the environment.
<img width="2560" height="828" alt="SAC_architecture_POMDP_v5" src="https://github.com/user-attachments/assets/6e00b9c8-15f5-4069-9cd2-6cc275cfd3ef" />


The SAC algorithm was developed in MATLAB to maintain compatibility with existing Simulink models and to allow for C code generation directly in MATLAB for future use in the aircraft's avionics.
The code was developed in consultation with the following sources:
1. OpenAI Spinning Up
2. K. Dally GitHub repo
3. Stable Baselines3
4. The original paper by Haarnoja: arXiv 1812.05905
5.  MATLAB's RL toolbox documentation
