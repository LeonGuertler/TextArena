#!/usr/bin/env python3
"""Test script to write a record to the Supabase user table."""

import os
import sys
from pathlib import Path

# Add the examples/fullstack_demo/backend directory to the path
backend_dir = Path(__file__).parent / "examples" / "fullstack_demo" / "backend"
sys.path.insert(0, str(backend_dir))

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

from supabase_client import get_supabase_user_manager


def test_write_user():
    """Test writing a user record to Supabase."""
    print("Testing Supabase user table write...")
    print(f"Supabase URL: {os.getenv('SUPABASE_URL')}")
    print(f"User table: {os.getenv('SUPABASE_USER_TABLE', 'users')}")
    print(f"Service Role Key present: {'Yes' if os.getenv('SUPABASE_SERVICE_ROLE_KEY') else 'No'}")
    print()
    print("NOTE: If you get 'permission denied for schema public', the table likely doesn't exist.")
    print("      Run the SQL in create_user_table.sql in your Supabase SQL Editor.")
    print("-" * 50)
    
    try:
        # Get the user manager
        user_manager = get_supabase_user_manager()
        print(f"✅ User manager created (table: {user_manager.table_name})")
        print()
        
        # Test data
        test_user_id = "test_user_123"
        test_name = "Test User"
        
        print(f"Attempting to write user record:")
        print(f"  user_id: {test_user_id}")
        print(f"  name: {test_name}")
        print()
        
        # Try to read from the table first to see if we can access it
        print("Testing table access (read)...")
        try:
            read_test = (
                user_manager.client.table(user_manager.table_name)
                .select("*")
                .limit(1)
                .execute()
            )
            print(f"✅ Table read successful! Found {len(read_test.data)} records")
            if read_test.data:
                print(f"  Sample record keys: {list(read_test.data[0].keys())}")
            print()
        except Exception as read_err:
            print(f"⚠️  Table read failed: {type(read_err).__name__}: {read_err}")
            print("   This suggests the table might not exist or permissions are incorrect.")
            print()
        
        # Try a simple direct insert first to see if it works
        print("Attempting direct insert...")
        import uuid
        test_uuid = str(uuid.uuid4())
        direct_payload = {
            "uuid": test_uuid,
            "user_id": test_user_id + "_direct",
            "name": test_name + " (Direct)",
        }
        
        try:
            direct_result = (
                user_manager.client.table(user_manager.table_name)
                .insert(direct_payload)
                .execute()
            )
            print("✅ Direct insert successful!")
            print(f"  Result: {direct_result.data}")
            print()
        except Exception as direct_err:
            print(f"⚠️  Direct insert failed: {type(direct_err).__name__}: {direct_err}")
            print()
        
        # Write the record (get_or_create will insert if it doesn't exist)
        print("Attempting insert via get_or_create_user_index...")
        result = user_manager.get_or_create_user_index(
            user_id=test_user_id,
            name=test_name
        )
        
        print("✅ Success! Record written/retrieved:")
        print(f"  UUID: {result['uuid']}")
        print(f"  Index: {result['index']}")
        print()
        
        # Verify by reading it back
        print("Verifying record by reading it back...")
        existing = (
            user_manager.client.table(user_manager.table_name)
            .select("*")
            .eq("uuid", result["uuid"])
            .limit(1)
            .execute()
        )
        
        if existing.data:
            print("✅ Verification successful! Record found:")
            for key, value in existing.data[0].items():
                print(f"  {key}: {value}")
        else:
            print("⚠️  Warning: Could not verify record (but insert seemed successful)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_write_user()
    sys.exit(0 if success else 1)

