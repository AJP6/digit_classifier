import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
from model import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

EPOCHS = 10

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

train_dataset = datasets.MNIST(
    root='./data', 
    train=True, download=True, 
    transform = transform 
)

test_dataset = datasets.MNIST(
    root='./data', 
    train=False, download=True, 
    transform = transform
)

def main(): 
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )

    model = Model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = torch.nn.CrossEntropyLoss()

    for e in range(EPOCHS):
        model.train()
        #batch loop 
        for batch, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()

            predictions = model(inputs)

            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            if (batch + 1) % 500 == 0: # every 500 batches
                print(f"  Epoch [{e+1}/{EPOCHS}], Batch [{batch+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print("\nTraining Finished! Running Final Evaluation...")

    #final tests
    with torch.no_grad(): 
        total_correct = 0
        total_checked = 0
        model.eval()
        for inputs, targets in test_loader: 
            inputs = inputs.to(DEVICE)
            #(batch_size, 10)
            targets = targets.to(DEVICE)

            #(batch_size, 10)
            predictions = model(inputs)
            _, digits = torch.max(predictions, dim=1) # ignores confidence and focuses on digit prediction

            total_correct += (digits == targets).sum().item()
            total_checked += targets.size(0)

        accuracy = 100 * (total_correct / total_checked) 
        print(f"Final Test Accuracy: {accuracy:.2f}%")


    torch.save(model.state_dict(), "digit_model.pth")
    print("Saved model weights to digit_model.pth")
    
if __name__ == "__main__": 
    main()