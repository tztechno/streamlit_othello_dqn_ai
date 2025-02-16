import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import pickle
import os


# Experience replay memory
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))


class OthelloGame:
    def __init__(self):
        self.board_size = 8
        self.reset()
    
    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        # Initial position
        center = self.board_size // 2
        self.board[center-1:center+1, center-1:center+1] = [[-1, 1], [1, -1]]
        self.current_player = 1  # 1 for black, -1 for white
        return self.get_state()
    
    def get_valid_moves(self):
        valid_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.is_valid_move(i, j):
                    valid_moves.append((i, j))
        return valid_moves
    
    def is_valid_move(self, row, col):
        if self.board[row, col] != 0:
            return False
            
        directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
        for dr, dc in directions:
            if self._would_flip(row, col, dr, dc):
                return True
        return False
    
    def _would_flip(self, row, col, dr, dc):
        r, c = row + dr, col + dc
        to_flip = []
        while 0 <= r < self.board_size and 0 <= c < self.board_size:
            if self.board[r, c] == 0:
                return False
            if self.board[r, c] == self.current_player:
                return len(to_flip) > 0
            to_flip.append((r, c))
            r, c = r + dr, c + dc
        return False
    
    def make_move(self, row, col):
        if not self.is_valid_move(row, col):
            return False
        
        self.board[row, col] = self.current_player
        directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
        pieces_flipped = 0
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            to_flip = []
            while 0 <= r < self.board_size and 0 <= c < self.board_size:
                if self.board[r, c] == 0:
                    break
                if self.board[r, c] == self.current_player:
                    for flip_r, flip_c in to_flip:
                        self.board[flip_r, flip_c] = self.current_player
                        pieces_flipped += 1
                    break
                to_flip.append((r, c))
                r, c = r + dr, c + dc
        
        self.current_player *= -1
        if not self.get_valid_moves():
            self.current_player *= -1
        
        return True
    
    def get_state(self):
        return self.board.copy()
    
    def get_winner(self):
        if self.get_valid_moves():
            return None
        
        black_count = np.sum(self.board == 1)
        white_count = np.sum(self.board == -1)
        
        if black_count > white_count:
            return 1
        elif white_count > black_count:
            return -1
        else:
            return 0


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 64)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 128 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DQN_Agent:
    def __init__(self, learning_rate=0.001, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)
        
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
    
    def get_action(self, state, valid_moves, training=True):
        if not valid_moves:
            return None
            
        if training and random.random() < self.epsilon:
            return random.choice(valid_moves)
            
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            
        valid_q_values = [(move, q_values[0][move[0] * 8 + move[1]].item()) 
                         for move in valid_moves]
        return max(valid_q_values, key=lambda x: x[1])[0]
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        experiences = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor([exp.state for exp in experiences]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor([exp.next_state for exp in experiences]).unsqueeze(1).to(self.device)
        actions = torch.LongTensor([[exp.action[0] * 8 + exp.action[1]] for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in experiences]).to(self.device)
        dones = torch.FloatTensor([exp.done for exp in experiences]).to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


class OthelloAI:
    def __init__(self):
        self.game = OthelloGame()
        self.ai = DQN_Agent(learning_rate=0.001, gamma=0.99)
        self.trained = False
    
    def train(self, num_episodes=30000, batch_size=32):
        print("Training AI...")
        for episode in range(num_episodes):
            state = self.game.reset()
            done = False
            
            while not done:
                valid_moves = self.game.get_valid_moves()
                if not valid_moves:
                    break
                    
                action = self.ai.get_action(state, valid_moves)
                if action:
                    self.game.make_move(*action)
                    next_state = self.game.get_state()
                    reward = self._calculate_reward(next_state)
                    done = self.game.get_winner() is not None
                    
                    self.ai.remember(state, action, reward, next_state, done)
                    self.ai.replay(batch_size)
                    
                    state = next_state
            
            if (episode + 1) % 1000 == 0:
                self.ai.update_target_network()
                print(f"Episode {episode + 1}/{num_episodes}, Epsilon: {self.ai.epsilon:.2f}")
        
        self.trained = True
        print("Training completed!")

##########

    def save_model(self, save_dir='models'):
        """Save the trained model and optimizer states"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save models and training state
        torch.save({
            'policy_net_state_dict': self.ai.policy_net.state_dict(),
            'target_net_state_dict': self.ai.target_net.state_dict(),
            'optimizer_state_dict': self.ai.optimizer.state_dict(),
            'epsilon': self.ai.epsilon,
            'memory': list(self.ai.memory)  # Convert deque to list for saving
        }, os.path.join(save_dir, 'othello_model.pth'))
        
        print(f"Model saved to {os.path.join(save_dir, 'othello_model.pth')}")

    ### new
    def load_model(self, model_path):
        """Load the trained model and optimizer states, handling both local files and URLs."""
    
        if model_path.startswith("http://") or model_path.startswith("https://"):
            # It's a URL, download it
            try:
                response = requests.get(model_path, stream=True)
                response.raise_for_status()
    
                filename = os.path.basename(model_path)
                filepath = filename  # Saves in the current directory. Change as needed.
    
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
    
                model_path = filepath  # Update model_path to the local file path
    
            except requests.exceptions.RequestException as e:
                raise Exception(f"Error downloading model from {model_path}: {e}")
            except Exception as e:
                raise Exception(f"Error saving model: {e}")
    
        # Now, model_path should be a local file path, whether it was originally a URL or not.
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model file found at {model_path}")
    
        # Load the saved state (your existing code)
        checkpoint = torch.load(model_path, map_location=self.ai.device)
    
        self.ai.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.ai.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.ai.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
        self.ai.epsilon = checkpoint['epsilon']
        self.ai.memory = deque(checkpoint['memory'], maxlen=10000)
    
        self.trained = True
    
        print(f"Model loaded from {model_path}")

    

    
##########
    
    def _calculate_reward(self, state):
        winner = self.game.get_winner()
        if winner is None:
            # Count pieces as intermediate reward
            return (np.sum(state == 1) - np.sum(state == -1)) * 0.01
        return 1 if winner == 1 else -1 if winner == -1 else 0
    
    def play_against_human(self, human_first=True):
        if not self.trained:
            print("Please train the AI first using the train() method.")
            return
        
        state = self.game.reset()
        current_player = 1 if human_first else -1
        
        while True:
            print("\nCurrent board:")
            self._print_board()
            
            valid_moves = self.game.get_valid_moves()
            if not valid_moves:
                break
            
            if current_player == 1:  # Human's turn
                print("Valid moves:", valid_moves)
                while True:
                    try:
                        row = int(input("Enter row (0-7): "))
                        col = int(input("Enter col (0-7): "))
                        if (row, col) in valid_moves:
                            self.game.make_move(row, col)
                            break
                        else:
                            print("Invalid move!")
                    except ValueError:
                        print("Invalid input!")
            else:  # AI's turn
                print("AI is thinking...")
                action = self.ai.get_action(state, valid_moves, training=False)
                if action:
                    print(f"AI moves to: {action}")
                    self.game.make_move(*action)
            
            state = self.game.get_state()
            current_player *= -1
        
        winner = self.game.get_winner()
        print("\nGame Over!")
        if winner == 1:
            print("Human wins!")
        elif winner == -1:
            print("AI wins!")
        else:
            print("It's a tie!")
    
    def _print_board(self):
        for i in range(self.game.board_size):
            for j in range(self.game.board_size):
                if self.game.board[i, j] == 1:
                    print("B", end=" ")
                elif self.game.board[i, j] == -1:
                    print("W", end=" ")
                else:
                    print(".", end=" ")
            print()


def train_first_moves(episodes=1000, batch_size=32):
    game = OthelloGame()
    ai = OthelloAI()
    
    for episode in range(episodes):
        state = game.reset()
        done = False
        
        # Only train on first move
        valid_moves = game.get_valid_moves()
        action = ai.get_action(state, valid_moves)
        
        if action:
            game.make_move(*action)
            next_state = game.get_state()
            reward = 1 if game.get_winner() == 1 else -1 if game.get_winner() == -1 else 0
            done = game.get_winner() is not None
            
            ai.remember(state, action, reward, next_state, done)
            ai.replay(batch_size)
        
        if (episode + 1) % 100 == 0:
            ai.update_target_network()
            print(f"Episode {episode + 1}/{episodes}, Epsilon: {ai.epsilon:.2f}")
    
    return ai

    
def train_second_moves(first_move_ai, episodes=1000, batch_size=32):
    game = OthelloGame()
    ai = OthelloAI()
    
    for episode in range(episodes):
        state = game.reset()
        
        # Make first move using first_move_ai
        valid_moves = game.get_valid_moves()
        first_action = first_move_ai.get_action(state, valid_moves)
        if first_action:
            game.make_move(*first_action)
        
        # Train on second move
        state = game.get_state()
        valid_moves = game.get_valid_moves()
        action = ai.get_action(state, valid_moves)
        
        if action:
            game.make_move(*action)
            next_state = game.get_state()
            reward = 1 if game.get_winner() == 1 else -1 if game.get_winner() == -1 else 0
            done = game.get_winner() is not None
            
            ai.remember(state, action, reward, next_state, done)
            ai.replay(batch_size)
        
        if (episode + 1) % 100 == 0:
            ai.update_target_network()
            print(f"Episode {episode + 1}/{episodes}, Epsilon: {ai.epsilon:.2f}")
    
    return ai


def play_against_human(ai):
    game = OthelloGame()
    state = game.reset()
    
    while True:
        # Print board
        for i in range(game.board_size):
            for j in range(game.board_size):
                if game.board[i, j] == 1:
                    print("B", end=" ")
                elif game.board[i, j] == -1:
                    print("W", end=" ")
                else:
                    print(".", end=" ")
            print()
        
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            break
        
        if game.current_player == 1:  # AI's turn
            action = ai.get_action(state, valid_moves)
            if action:
                print(f"AI moves to: {action}")
                game.make_move(*action)
        else:  # Human's turn
            print("Valid moves:", valid_moves)
            while True:
                try:
                    row = int(input("Enter row (0-7): "))
                    col = int(input("Enter col (0-7): "))
                    if (row, col) in valid_moves:
                        game.make_move(row, col)
                        break
                    else:
                        print("Invalid move!")
                except ValueError:
                    print("Invalid input!")
        
        state = game.get_state()
    
    winner = game.get_winner()
    if winner == 1:
        print("AI wins!")
    elif winner == -1:
        print("Human wins!")
    else:
        print("It's a tie!")


def make_move(game, ai, row, col):
    """
    Makes a move in the game. Returns (is_valid, game_state, valid_moves, winner)
    game_state is the board after the move
    valid_moves are the possible moves for the next player
    winner is None if game is ongoing, 1 for AI win, -1 for human win, 0 for tie
    """
    if not game.is_valid_move(row, col):
        return False, game.get_state(), game.get_valid_moves(), None
        
    game.make_move(row, col)
    
    # AI's turn
    valid_moves = game.get_valid_moves()
    if valid_moves:
        action = ai.get_action(game.get_state(), valid_moves)
        if action:
            game.make_move(*action)
    
    return True, game.get_state(), game.get_valid_moves(), game.get_winner()


def get_game_state():
    """
    Creates a new game and returns the initial state and valid moves
    """
    game = OthelloGame()
    state = game.reset()
    return game, state, game.get_valid_moves()


def print_board(board):
    """
    Returns a string representation of the board
    """
    board_str = ""
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == 1:
                board_str += "B "
            elif board[i][j] == -1:
                board_str += "W "
            else:
                board_str += ". "
        board_str += "\n"
    return board_str


def play(r,l):
    is_valid, new_state, next_moves, winner = make_move(game, ai, r,l)
    if is_valid:
        print("After move:")
        print(print_board(new_state))
        if winner is not None:
            if winner == 1:
                print("AI wins!")
            elif winner == -1:
                print("Human wins!")
            else:
                print("It's a tie!")
    else:
        print("Invalid move!")

        
def play_game(player_color, white_ai=None, black_ai=None):
    game, state, valid_moves = get_game_state()
    print("\nInitial board:")
    print(print_board(state))
    print("Valid moves:", valid_moves)

    ai = white_ai if player_color == "B" else black_ai
    while valid_moves:  # Game continues while there are valid moves
        if game.current_player == player_color:
            move = get_player_move(valid_moves)  # Player's move
        else:
            move = ai.get_best_move(state, valid_moves)  # AI's move
            print(f"AI moves: {move}")

        game.make_move(move)
        state, valid_moves = game.get_state()
        print(print_board(state))
        print("Valid moves:", valid_moves)

    print("Game over!")
    winner = game.get_winner()  # Check the winner (if method exists)
    print(f"Winner: {winner}")


def ai_vs_ai(white_ai, black_ai):
    game, state, valid_moves = get_game_state()
    print("\nStarting AI vs AI match...")

    while valid_moves:  # Continue while there are valid moves
        ai = white_ai if game.current_player == "W" else black_ai
        move = ai.get_best_move(state, valid_moves)
        game.make_move(move)
        state, valid_moves = game.get_state()
        print(print_board(state))

    print("AI match over!")
    winner = game.get_winner()
    print(f"Winner: {winner}")



def train_and_save():
    game = OthelloAI()
    game.train(num_episodes=100)
    game.save_model()

def load_and_play():
    game = OthelloAI()
    game.load_model("https://huggingface.co/stpete2/dqn_othello_20250216/resolve/main/othello_model.pth")   
    game.play_against_human(human_first=False)
    
