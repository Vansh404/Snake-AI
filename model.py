import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
	def __init__(self,input_size,hidden_size,output_size):
		super().__init__()
		self.linear1=nn.Linear(input_size,hidden_size)
		self.linear2=nn.Linear(hidden_size,output_size)

	def forward(self,x):
		x = F.relu(self.linear1(x)) #Using the ReLu activation function
		x=self.linear2(x)
		return x

	def save(self,file_name='torch_model.pth'):
		model_folder_path='./model'
		if not os.path.exists(model_folder_path):
			os.makedirs(model_folder_path)

		file_name=os.path.join(model_folder_path,file_name)
		torch.save(self.state_dict(),file_name)


class QTrainer:
	def __init__(self,model,lr,gamma):
		self.lr=lr
		self.gamma=gamma
		self.model=model
		self.optimizer=optim.Adam(model.parameters(),lr=self.lr) #using the Adam optimiser
		self.criterion=nn.MSELoss()#the loss function

	def train_step(self,old_state,action,reward,new_state,done):
		
		old_state = torch.tensor(old_state, dtype=torch.float)
		new_state = torch.tensor(new_state, dtype=torch.float)
		action = torch.tensor(action, dtype=torch.long)
		reward = torch.tensor(reward, dtype=torch.float)

		if len(old_state.shape) == 1:
			old_state = torch.unsqueeze(old_state,0)
			new_state = torch.unsqueeze(new_state,0)
			action = torch.unsqueeze(action,0)
			reward = torch.unsqueeze(reward,0)
			done=(done,)


		pred =  self.model(old_state) #get predicted values of Q with current state
		target = pred.clone()

		for idx in range(len(done)):
			Q_new=reward[idx]
			
			if not done[idx]:
				Q_new=reward[idx]+self.gamma * torch.max(self.model(new_state[idx]))

			target[idx][torch.argmax(action).item()]=Q_new #choosing the action with the highest reward

		self.optimizer.zero_grad()
		loss=self.criterion(target,pred)
		loss.backward()#implement backpropagation to update weights

		self.optimizer.step()


