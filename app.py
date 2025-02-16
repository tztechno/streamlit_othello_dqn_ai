import streamlit as st
import numpy as np
from othello_ai import OthelloAI, OthelloGame

def init_session_state():
    if 'game' not in st.session_state:
        st.session_state.game = OthelloGame()
    if 'ai' not in st.session_state:
        st.session_state.ai = OthelloAI()
        # Load the pre-trained model
        st.session_state.ai.load_model("https://huggingface.co/stpete2/dqn_othello_20250216/resolve/main/othello_model.pth")
    if 'board' not in st.session_state:
        st.session_state.board = st.session_state.game.get_state()
    if 'game_over' not in st.session_state:
        st.session_state.game_over = False

def reset_game():
    st.session_state.game = OthelloGame()
    st.session_state.board = st.session_state.game.get_state()
    st.session_state.game_over = False

def make_move(row, col):
    if not st.session_state.game_over:
        # Human move
        if st.session_state.game.make_move(row, col):
            st.session_state.board = st.session_state.game.get_state()
            
            # Check if game is over after human move
            if not st.session_state.game.get_valid_moves():
                st.session_state.game_over = True
                return
            
            # AI move
            valid_moves = st.session_state.game.get_valid_moves()
            if valid_moves:
                action = st.session_state.ai.get_action(st.session_state.board, valid_moves, training=False)
                if action:
                    st.session_state.game.make_move(*action)
                    st.session_state.board = st.session_state.game.get_state()
            
            # Check if game is over after AI move
            if not st.session_state.game.get_valid_moves():
                st.session_state.game_over = True

def main():
    st.title("Othello Game")
    st.write("Play against AI (You are Black, AI is White)")
    
    init_session_state()
    
    # Reset button
    if st.button("New Game"):
        reset_game()
    
    # Create the game board display
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display the board as a grid of buttons
        for i in range(8):
            cols = st.columns(8)
            for j in range(8):
                with cols[j]:
                    piece = st.session_state.board[i][j]
                    if piece == 1:  # Black
                        button_label = "⚫"
                    elif piece == -1:  # White
                        button_label = "⚪"
                    else:
                        button_label = " "
                    
                    # Make the button disabled if the position is already occupied
                    if piece != 0:
                        st.button(button_label, key=f"btn_{i}_{j}", disabled=True)
                    else:
                        # Only enable valid moves
                        is_valid = st.session_state.game.is_valid_move(i, j)
                        if st.button(button_label, key=f"btn_{i}_{j}", disabled=not is_valid):
                            make_move(i, j)
    
    with col2:
        # Display game status
        black_count = np.sum(st.session_state.board == 1)
        white_count = np.sum(st.session_state.board == -1)
        
        st.write("Score:")
        st.write(f"Black (You): {black_count}")
        st.write(f"White (AI): {white_count}")
        
        if st.session_state.game_over:
            winner = st.session_state.game.get_winner()
            if winner == 1:
                st.success("You win!")
            elif winner == -1:
                st.error("AI wins!")
            else:
                st.info("It's a tie!")
        else:
            valid_moves = st.session_state.game.get_valid_moves()
            if valid_moves:
                st.write("Valid moves:", valid_moves)

if __name__ == "__main__":
    main()
    
