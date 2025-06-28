import numpy as np
import time

def run_simulation(steps=10):
    position = 0.0
    for step in range(steps):
        position += np.random.randn()
        print(f"Step {step}: {position:.3f}")
        time.sleep(0.1)
    print("Simulation complete!")

if __name__ == "__main__":
    run_simulation()
