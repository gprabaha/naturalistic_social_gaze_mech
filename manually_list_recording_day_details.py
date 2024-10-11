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

def main():
    recording_days = []
    last_m1 = None
    last_m2 = None
    
    print("Enter details for each recording day. Type 'exit' to stop.")

    while True:
        session_name = input("Enter session name (or 'exit' to finish): ").strip()
        if session_name.lower() == 'exit':
            break
        
        m1 = input_or_reuse("Enter m1 data", last_m1)
        m2 = input_or_reuse("Enter m2 data", last_m2)
        
        add_recording_day(recording_days, session_name, m1, m2)
        
        # Update the last m1 and m2
        last_m1 = m1
        last_m2 = m2
    
    print("\nRecording days list created:")
    for day in recording_days:
        print(day)

if __name__ == "__main__":
    main()
