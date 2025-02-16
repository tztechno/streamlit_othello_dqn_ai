
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import requests
import os

# Configure page
st.set_page_config(page_title="Othello AI", layout="wide")

# Constants
BOARD_SIZE = 8
DIRECTIONS = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]

# Game logic
def create_board():
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    center = BOARD_SIZE // 2
    board[center-1:center+1, center-1:center+1] = [[-1, 1], [1, -1]]
    return board

def is_valid_move(board, row, col, player):
    if board[row, col] != 0:
        return False
    
    for dr, dc in DIRECTIONS:
        r, c = row + dr, col + dc
        to_flip = []
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            if board[r, c] == 0:
                break
            if board[r, c] == player:
                if to_flip:
                    return True
                break
            to_flip.append((r, c))
            r, c = r + dr, c + dc
    return False

def get_valid_moves(board, player):
    return [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) 
            if is_valid_move(board, i, j, player)]

def make_move(board, row, col, player):
    if not is_valid_move(board, row, col, player):
        return board, False
    
    board = board.copy()
    board[row, col] = player
    
    for dr, dc in DIRECTIONS:
        r, c = row + dr, col + dc
        to_flip = []
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            if board[r, c] == 0:
                break
            if board[r, c] == player:
                for flip_r, flip_c in to_flip:
                    board[flip_r, flip_c] = player
                break
            to_flip.append((r, c))
            r, c = r + dr, c + dc
    
    return board, True

def get_winner(board):
    black_count = np.sum(board == 1)
    white_count = np.sum(board == -1)
    
    if black_count > white_count:
        return 1
    elif white_count > black_count:
        return -1
    return 0

# AI Model
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * BOARD_SIZE * BOARD_SIZE, 64)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 64 * BOARD_SIZE * BOARD_SIZE)
        return self.fc(x)

# Initialize session state
if 'board' not in st.session_state:
    st.session_state.board = create_board()
if 'current_player' not in st.session_state:
    st.session_state.current_player = 1
if 'model' not in st.session_state:
    st.session_state.model = DQN()
    try:
        model_url = "https://huggingface.co/stpete2/dqn_othello_20250216/resolve/main/othello_model.pth"
        response = requests.get(model_url)
        with open("model.pth", "wb") as f:
            f.write(response.content)
        checkpoint = torch.load("model.pth", map_location='cpu')
        st.session_state.model.load_state_dict(checkpoint['policy_net_state_dict'])
        st.session_state.model.eval()
    except Exception as e:
        st.error(f"Error loading model: {e}")

def get_ai_move(board):
    valid_moves = get_valid_moves(board, -1)
    if not valid_moves:
        return None
    
    state = torch.FloatTensor(board).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        q_values = st.session_state.model(state)
    
    valid_q_values = [(move, q_values[0][move[0] * 8 + move[1]].item()) 
                      for move in valid_moves]
    return max(valid_q_values, key=lambda x: x[1])[0]

def handle_move(row, col):
    if st.session_state.current_player != 1:
        return
        
    # Human move
    board, valid = make_move(st.session_state.board, row, col, 1)
    if valid:
        st.session_state.board = board
        st.session_state.current_player = -1
        
        # AI move
        ai_move = get_ai_move(st.session_state.board)
        if ai_move:
            board, _ = make_move(st.session_state.board, *ai_move, -1)
            st.session_state.board = board
            st.session_state.current_player = 1

def main():
    st.title("Othello Game")
    st.write("You play as Black (⚫), AI plays as White (⚪)")
    
    # Reset button
    if st.button("New Game"):
        st.session_state.board = create_board()
        st.session_state.current_player = 1
    
    # Game board
    for i in range(BOARD_SIZE):
        cols = st.columns(BOARD_SIZE)
        for j in range(BOARD_SIZE):
            with cols[j]:
                piece = st.session_state.board[i][j]
                label = "⚫" if piece == 1 else "⚪" if piece == -1 else " "
                if st.button(label, key=f"btn_{i}_{j}"):
                    handle_move(i, j)
    
    # Game status
    st.write("---")
    black_count = np.sum(st.session_state.board == 1)
    white_count = np.sum(st.session_state.board == -1)
    
    st.write(f"Score - Black: {black_count}, White: {white_count}")
    
    valid_moves = get_valid_moves(st.session_state.board, st.session_state.current_player)
    if not valid_moves:
        winner = get_winner(st.session_state.board)
        if winner == 1:
            st.success("You win!")
        elif winner == -1:
            st.error("AI wins!")
        else:
            st.info("It's a tie!")

if __name__ == "__main__":
    main()


