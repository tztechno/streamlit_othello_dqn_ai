

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import requests
import os
import io

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



class OthelloAI:
    def __init__(self):
        self.game = OthelloGame()
        self.ai = DQN_Agent(learning_rate=0.001, gamma=0.99)
        self.trained = False
    
    def load_model(self, model_path):
        try:
            if model_path.startswith("http://") or model_path.startswith("https://"):
                response = requests.get(model_path, stream=True)
                response.raise_for_status()
                
                model_bytes = io.BytesIO()
                for chunk in response.iter_content(chunk_size=8192):
                    model_bytes.write(chunk)
                model_bytes.seek(0)
                
                # Experience クラスを安全なグローバルとして追加
                with torch.serialization.safe_globals([Experience]):
                    checkpoint = torch.load(model_bytes, map_location=self.ai.device)
            else:
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"No model file found at {model_path}")
                checkpoint = torch.load(model_path, map_location=self.ai.device)

            # checkpoint が None でないことを確認
            if checkpoint is None:
                raise ValueError("Loaded checkpoint is None")

            # 必要なキーが存在することを確認
            required_keys = ['policy_net_state_dict', 'target_net_state_dict', 
                           'optimizer_state_dict', 'epsilon', 'memory']
            for key in required_keys:
                if key not in checkpoint:
                    raise KeyError(f"Missing required key in checkpoint: {key}")

            # モデルの状態を読み込む
            self.ai.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.ai.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.ai.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.ai.epsilon = checkpoint['epsilon']
            self.ai.memory = deque(checkpoint['memory'], maxlen=10000)
            self.trained = True
            
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.trained = False
            return False


if 'game' not in st.session_state:
    st.session_state.game = OthelloGame()
    st.session_state.ai = OthelloAI()
    if st.session_state.ai.load_model("https://huggingface.co/stpete2/dqn_othello_20250216/resolve/main/othello_model.pth"):
        st.success("AI model loaded successfully!")
    else:
        st.error("Failed to load AI model. Using untrained model.")


def handle_move(i, j):
    """Handle a move at position (i, j)"""
    current_player = st.session_state.game.current_player
    human_first = st.session_state.player_color == "Black (First)"
    
    if (human_first and current_player == 1) or (not human_first and current_player == -1):
        if st.session_state.game.is_valid_move(i, j):
            # Make human move
            st.session_state.game.make_move(i, j)
            st.session_state.last_move = (i, j)
            
            # AI's turn
            valid_moves = st.session_state.game.get_valid_moves()
            if valid_moves:
                action = st.session_state.ai.ai.get_action(
                    st.session_state.game.get_state(),
                    valid_moves,
                    training=False
                )
                if action:
                    st.session_state.game.make_move(*action)
                    st.session_state.ai_last_move = action
            
            # Force rerun to update the board
            st.rerun()

# Streamlit interface
st.title("Play Othello Against AI")

# Player selection
if 'player_color' not in st.session_state:
    st.session_state.player_color = "Black (First)"

player_color = st.radio("Choose your color:", ["Black (First)", "White (Second)"], key="player_color")
human_first = player_color == "Black (First)"

# Reset game button
if st.button("Reset Game"):
    st.session_state.game = OthelloGame()
    st.session_state.last_move = None
    st.session_state.ai_last_move = None
    if not human_first:  # If AI goes first
        valid_moves = st.session_state.game.get_valid_moves()
        if valid_moves:
            action = st.session_state.ai.ai.get_action(
                st.session_state.game.get_state(),
                valid_moves,
                training=False
            )
            if action:
                st.session_state.game.make_move(*action)
                st.session_state.ai_last_move = action

# Initialize last move trackers if not exists
if 'last_move' not in st.session_state:
    st.session_state.last_move = None
if 'ai_last_move' not in st.session_state:
    st.session_state.ai_last_move = None

# Create a compact game board display
cols = st.columns([1, 2, 1])  # Create three columns for layout
with cols[1]:  # Use the middle column for the board
    board = st.session_state.game.get_state()
    for i in range(8):
        cols_board = st.columns(8)
        for j in range(8):
            with cols_board[j]:
                # Determine the piece to display and styling
                if board[i][j] == 1:
                    piece = "⚫"
                elif board[i][j] == -1:
                    piece = "⚪"
                else:
                    piece = "·"
                
                # Create a button for each cell
                st.button(
                    piece, 
                    key=f"cell_{i}_{j}", 
                    on_click=handle_move, 
                    args=(i, j),
                    help=f"Position ({i}, {j})",
                    use_container_width=True
                )

# Game status and information
col1, col2 = st.columns(2)

with col1:
    # Score
    board = st.session_state.game.get_state()
    black_count = np.sum(board == 1)
    white_count = np.sum(board == -1)
    st.write(f"Score - Black: {black_count}, White: {white_count}")
    
    # Current player
    current_player = "Black" if st.session_state.game.current_player == 1 else "White"
    st.write(f"Current player: {current_player}")

with col2:
    # Game status
    winner = st.session_state.game.get_winner()
    if winner is not None:
        if winner == 1:
            st.success("Black wins!")
        elif winner == -1:
            st.success("White wins!")
        else:
            st.info("It's a tie!")

    # Valid moves
    valid_moves = st.session_state.game.get_valid_moves()
    if valid_moves:
        st.write("Valid moves:", valid_moves)
    else:
        if winner is None:
            st.write("No valid moves available. Turn passes.")




