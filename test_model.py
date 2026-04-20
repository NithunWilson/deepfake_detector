"""
Test the trained model to verify it works properly
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from train_proper import ImprovedModel, BalancedDataset
from torch.utils.data import DataLoader

def test_model_behavior():
    print("Testing model behavior...")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedModel().to(device)
    
    # Try to load trained weights
    try:
        checkpoint = torch.load('models/deepfake_model_final.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded trained model")
    except:
        print("No trained model found. Testing with random weights")
    
    model.eval()
    
    # Create test samples
    test_dataset = BalancedDataset(num_samples=20, seq_length=50)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Test predictions
    real_predictions = []
    fake_predictions = []
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            
            for j in range(len(labels)):
                if labels[j] == 0:  # Real
                    real_predictions.append(probabilities[j, 0].item())  # Real probability
                else:  # Fake
                    fake_predictions.append(probabilities[j, 1].item())  # Fake probability
    
    # Analyze results
    print(f"\nReal videos (should have high Real probability):")
    print(f"  Average Real probability: {np.mean(real_predictions):.3f}")
    print(f"  Min/Max: {np.min(real_predictions):.3f}/{np.max(real_predictions):.3f}")
    
    print(f"\nFake videos (should have high Fake probability):")
    print(f"  Average Fake probability: {np.mean(fake_predictions):.3f}")
    print(f"  Min/Max: {np.min(fake_predictions):.3f}/{np.max(fake_predictions):.3f}")
    
    # Plot distributions
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(real_predictions, bins=20, alpha=0.7, label='Real Videos', color='green')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Boundary')
    plt.xlabel('Real Probability')
    plt.ylabel('Count')
    plt.title('Real Videos - Should be > 0.5')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(fake_predictions, bins=20, alpha=0.7, label='Fake Videos', color='red')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Boundary')
    plt.xlabel('Fake Probability')
    plt.ylabel('Count')
    plt.title('Fake Videos - Should be > 0.5')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_test_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Check if model is working
    real_avg = np.mean(real_predictions)
    fake_avg = np.mean(fake_predictions)
    
    if real_avg > 0.6 and fake_avg > 0.6:
        print("\n✓ Model is working correctly!")
        print("  - Real videos correctly identified as Real")
        print("  - Fake videos correctly identified as Fake")
    elif real_avg > 0.6 and fake_avg < 0.4:
        print("\n⚠ Model might be reversed (predicting opposite)")
    elif real_avg < 0.4 and fake_avg < 0.4:
        print("\n✗ Model is not confident")
    elif real_avg > 0.6 and fake_avg > 0.6:
        print("\n⚠ Model might be predicting both classes as one type")
    else:
        print("\n? Model behavior unclear")

if __name__ == "__main__":
    test_model_behavior()
