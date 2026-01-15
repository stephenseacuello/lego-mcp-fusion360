# ADR-004: Physics-Informed Neural Networks for Digital Twin

## Status

Accepted

## Date

2026-01-08

## Context

Traditional digital twins in manufacturing face challenges:

1. **Pure Data-Driven Models**: Require massive training data, extrapolate poorly
2. **Pure Physics Models**: Computationally expensive, hard to calibrate
3. **Real-Time Constraints**: Manufacturing needs sub-second predictions
4. **Generalization**: Models must work across product variations

LEGO brick manufacturing has well-understood physics:
- Thermal: Injection molding heat transfer
- Structural: Clutch power, stud interference
- Fluid: Plastic melt flow in mold

## Decision

We will implement **Physics-Informed Neural Networks (PINNs)** that embed physical laws into the neural network architecture:

### 1. Architecture

The loss function combines:
- **Data Loss**: MSE between predictions and measurements
- **Physics Loss**: PDE residuals (heat equation, stress-strain)
- **Boundary Loss**: Enforce known boundary conditions

```
L_total = L_data + lambda_physics * L_physics + lambda_bc * L_boundary
```

### 2. Implemented Models

**Thermal PINN** (`thermal_twin`):
- Predicts temperature distribution during injection molding
- Embeds heat equation: dT/dt = alpha * nabla^2(T)
- Inputs: mold geometry, material properties, process parameters
- Outputs: 4D temperature field (x, y, z, t)

**Structural PINN** (`structural_twin`):
- Predicts clutch power and stress distribution
- Embeds linear elasticity equations
- Predicts stud interference fit quality
- Outputs: stress, strain, clutch force

**Cooling PINN** (`cooling_twin`):
- Predicts optimal cooling channel layout
- Embeds convective heat transfer
- Outputs: recommended cooling parameters

### 3. Training Strategy

- **Transfer Learning**: Pre-train on simulation data
- **Online Learning**: Fine-tune with real measurements
- **Uncertainty Quantification**: Monte Carlo dropout for confidence

### Implementation

- `dashboard/services/digital_twin/pinn_model.py`
- `dashboard/services/digital_twin/twin_ontology.py`

## Consequences

### Positive

- **Data Efficient**: Works with 100x less data than pure ML
- **Physically Consistent**: Predictions obey conservation laws
- **Extrapolation**: Generalizes to unseen conditions
- **Interpretable**: Physics terms provide insight
- **Real-Time**: Once trained, inference is fast (~10ms)

### Negative

- **Complex Training**: Must balance multiple loss terms
- **Domain Knowledge**: Requires physics expertise
- **Limited to Known Physics**: Can't capture unknown phenomena
- **Hyperparameter Sensitivity**: lambda values affect convergence

### Risks

- Physics model doesn't match real system
- Training instability with complex PDEs
- Computational cost of training

### Mitigations

- Validate against FEA simulations
- Use adaptive loss weighting
- Transfer learning from physics simulations
- Regular calibration with real measurements

## Architecture Diagram

```
              +------------------+
              |   Sensor Data    |
              |   (T, P, etc.)   |
              +--------+---------+
                       |
              +--------v---------+
              |    Input Layer   |
              |  (x,y,z,t,params)|
              +--------+---------+
                       |
              +--------v---------+
              |   Hidden Layers  |
              |   (ResNet-style) |
              +--------+---------+
                       |
        +------+-------+-------+------+
        |      |               |      |
   +----v---+  |          +----v----+ |
   |L_data  |  |          |L_physics| |
   |   MSE  |  |          |   PDE   | |
   +--------+  |          +---------+ |
               |                      |
          +----v----+           +-----v-----+
          |L_boundary|          | L_initial |
          |   BCs   |           |    ICs    |
          +---------+           +-----------+
```

## Implementation Notes

```python
# Create thermal digital twin
from dashboard.services.digital_twin.pinn_model import create_thermal_twin

twin = create_thermal_twin(brick_type="4x2", material="ABS")

# Predict temperature field
result = twin.predict(
    x=np.linspace(0, 0.032, 10),
    y=np.linspace(0, 0.016, 8),
    z=np.linspace(0, 0.01, 5),
    t=np.array([0, 5, 10, 15, 20]),
    mold_temp=80,
    melt_temp=220,
)

# Get prediction with uncertainty
prediction = twin.predict_with_uncertainty(
    inputs, n_samples=100
)
print(f"Temperature: {prediction.mean:.1f} +/- {prediction.std:.2f} C")
```

## Validation Results

| Metric | Pure NN | Pure FEA | PINN |
|--------|---------|----------|------|
| RMSE (training) | 2.1 C | 0.5 C | 1.2 C |
| RMSE (test) | 8.5 C | 0.5 C | 1.8 C |
| Extrapolation error | 25% | 2% | 5% |
| Inference time | 5ms | 300s | 10ms |
| Training data needed | 100K | 0 | 1K |

## References

- [Raissi et al. "Physics-Informed Neural Networks" (2019)](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125)
- [ISO 23247: Digital Twin Framework for Manufacturing](https://www.iso.org/standard/75066.html)
- [DeepXDE: Physics-Informed Deep Learning Library](https://github.com/lululxvi/deepxde)
