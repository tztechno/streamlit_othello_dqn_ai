

import streamlit as st
import numpy as np
import torch
from collections import deque
import requests
import os
import io

# [Previous class definitions for OthelloGame, DQN, DQN_Agent, and OthelloAI go here]

# Initialize session state
if 'game' not in st.session_state:
    st.session_state.game = OthelloGame()
    st.session_state.ai = OthelloAI()
    try:
        st.session_state.ai.load_model("https://huggingface.co/stpete2/dqn_othello_20250216/resolve/main/othello_model.pth")
        st.success("AI model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")

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
                
                # Highlight last moves
                button_style = ""
                if (i, j) == st.session_state.last_move:
                    button_style = "background-color: #ffeb3b;"
                elif (i, j) == st.session_state.ai_last_move:
                    button_style = "background-color: #90caf9;"
                
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



