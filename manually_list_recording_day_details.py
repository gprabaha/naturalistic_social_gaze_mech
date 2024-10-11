import pickle

def input_or_reuse(prompt, default_value=None):
    """Helper function to allow reusing the last entered value or entering a new one."""
    user_input = input(f"{prompt} [{default_value}]: ").strip()
    return user_input if user_input else default_value

def add_recording_day(recording_days, session_name, m1=None, m2=None):
    """Adds a new recording day to the list."""
    recording_days.append({
        'session_name': session_name,
        'm1': m1,
        'm2': m2
    })

def save_recording_days(recording_days, filename="recording_days.pkl"):
    """Save the recording days list to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(recording_days, f)
    print(f"Recording days saved to {filename}")

def load_recording_days(filename="recording_days.pkl"):
    """Load the recording days list from a pickle file, if it exists."""
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return []

def main():
    recording_days = load_recording_days()
    last_m1 = None
    last_m2 = None
    
    print("Enter details for each recording day. Type 'exit' to stop.")
    
    while True:
        session_name = input("Enter session name (or 'exit' to finish): ").strip()
        if session_name.lower() == 'exit':
            break
        
        m1 = input_or_reuse("Enter m1 data", last_m1)
        m2 = input_or_reuse("Enter m2 data", last_m2)
        
        # Show entered data and ask for confirmation
        print(f"\nYou entered:\nSession: {session_name}\nm1: {m1}\nm2: {m2}")
        confirm = input("Is this correct? (yes/no): ").strip().lower()
        
        if confirm == 'yes':
            add_recording_day(recording_days, session_name, m1, m2)
            # Update the last m1 and m2
            last_m1 = m1
            last_m2 = m2
        else:
            print("Entry discarded. Please re-enter the details.")
    
    # Save the list before exiting
    save_recording_days(recording_days)
    print("\nRecording days list created:")
    for day in recording_days:
        print(day)

if __name__ == "__main__":
    main()
