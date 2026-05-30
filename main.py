from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# Initialize the Server
app = FastAPI(title="Spider V2 Backend")

# Allow React to talk to Python securely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (localhost:5173)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATA MODELS (What React sends to Python) ---
class Point(BaseModel):
    angle: float
    height: float
    health_ratio: float

class ContourRequest(BaseModel):
    points: List[Point]
    tank_height: float
    tank_radius: float
    resolution: int = 100

# --- THE MACHINE LEARNING ENGINE ---
@app.post("/api/generate-contour")
def generate_contour(req: ContourRequest):
    res = req.resolution
    
    # If the user hits execute with no data, return a blank red tank safely
    if not req.points:
        return {"resolution": res, "health_map": [0.0] * (res * res)}

    # 1. Prepare Training Data
    X_train = []
    y_train = []
    
    for p in req.points:
        # Convert cylindrical (Angle, Height) to 3D Cartesian (X, Y, Z)
        # This teaches the AI that 359 degrees and 1 degree are touching!
        rad = np.radians(p.angle)
        x = req.tank_radius * np.cos(rad)
        z = req.tank_radius * np.sin(rad)
        y = p.height
        
        X_train.append([x, y, z])
        y_train.append(p.health_ratio)
        
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # 2. Configure the AI (Gaussian Process Regressor with Matern Kernel)
    # length_scale dictates how far a single rust spot spreads its color
    kernel = Matern(length_scale=3.0, nu=1.5)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.01, normalize_y=False)
    
    # Train the AI on the inspector's data
    gpr.fit(X_train, y_train)

    # 3. Prepare the Blank Canvas (Resolution Grid)
    # Generate exact heights from the top of the tank down to the floor
    heights = np.linspace(req.tank_height, 0, res)
    angles = np.linspace(0, 360, res)
    
    X_predict = []
    for h in heights:
        for a in angles:
            rad = np.radians(a)
            x = req.tank_radius * np.cos(rad)
            z = req.tank_radius * np.sin(rad)
            X_predict.append([x, h, z])
            
    X_predict = np.array(X_predict)

    # 4. Predict the entire 10,000 pixel surface of the tank instantly!
    y_pred, _ = gpr.predict(X_predict, return_std=True)
    
    # 5. Safety Formatting
    # Ensure no weird negative thicknesses occur from math overshoots
    y_pred = np.clip(y_pred, 0.0, 3.0) 
    
    # Send the huge array of predictions back to React to be painted
    return {
        "resolution": res,
        "health_map": y_pred.tolist()
    }
