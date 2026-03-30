import torch
import torch.nn as nn
from tqdm import tqdm
from src.model import PlantDiseaseModel
from src.dataset import get_data_loader

def train_model(model, train_loader, val_loader, epochs = 10, lr = 1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')

    ## Modeling
    for epoch in range(epochs):
        ## Training

        model.train()
        train_loss = 0

        for images, labels in tqdm(train_loader, desc = f"Epoch {epoch+1}/{epochs} - Training"):
           images, labels = images.to(device), labels.to(device)

           outputs = model(images)
           loss = criterion(outputs, labels)

           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

           train_loss += loss.item()

        
        ## Validation
        model.eval()
        val_loss = 0
        correct = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc = f"Epoch {epoch+1}/{epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                val_loss += criterion(outputs, labels).item()

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * (correct / len(val_loader.dataset))

        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        ## Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'models/best_model.pth')

if __name__ == "__main__":
    train_loader, val_loader, class_names = get_data_loader('data')

    #Instantiate moodel
    model = PlantDiseaseModel(num_classes=len(class_names))

    train_model(model, train_loader, val_loader)