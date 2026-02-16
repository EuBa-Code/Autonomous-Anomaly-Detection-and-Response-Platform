"""
Test Feast Feature Store Functionality

This script demonstrates how to use the Feast API for:
1. Getting online features
2. Pushing streaming features
3. Materializing batch features
"""

if __name__ == '__main__':

    import pandas as pd
    from datetime import datetime, timedelta
    from feast import FeatureStore

    # Initialize feature store (repo_path must match feature_store.yaml location)
    store = FeatureStore(repo_path="/feature_store_service")

    print("=" * 80)
    print("FEAST FEATURE STORE API TESTS")
    print("=" * 80)

    # ============================================================================
    # TEST 1: Get Online Features by Feature Names
    # ============================================================================
    print("\n[TEST 1] Getting online features by feature names...")

    try:
        features = store.get_online_features(
            features=[
                "machine_features:Current_L1",
                "machine_features:Vibration_mm_s",
                "machine_features:Motor_RPM",
                "machine_features:Water_Temp_C"
            ],
            entity_rows=[
                {"Machine_ID": 1},  
                {"Machine_ID": 2},
                {"Machine_ID": 3}
            ]
        ).to_dict()
        
        print("✓ Successfully retrieved features")
        print(f"Features for {len(features['Machine_ID'])} machines:")
        for key, values in features.items():
            print(f"  {key}: {values}")
            
    except Exception as e:
        print(f"✗ Error: {e}")

    # ============================================================================
    # TEST 2: Get Online Features using Feature Service
    # ============================================================================
    print("\n[TEST 2] Getting online features using feature service...")

    try:
        # FIXED: Get the feature service object first, then pass it to get_online_features
        feature_service = store.get_feature_service("machine_anomaly_service_v1")
        
        features = store.get_online_features(
            features=feature_service,  # Pass the feature service object, not a string
            entity_rows=[
                {"Machine_ID": 1},
                {"Machine_ID": 2}
            ]
        ).to_dict()
        
        print("✓ Successfully retrieved features via feature service")
        print(f"Number of features returned: {len(features)}")
        print(f"Feature names: {list(features.keys())}")
        
    except Exception as e:
        print(f"✗ Error: {e}")

    # ============================================================================
    # TEST 3: Push Streaming Features
    # ============================================================================
    print("\n[TEST 3] Pushing streaming features to online store...")

    try:
        # Create sample streaming data
        now = datetime.now()
        streaming_data = pd.DataFrame({
            "Machine_ID": [1, 2, 3],
            "event_timestamp": [now, now, now],  # FIXED: Changed from "timestamp" to "event_timestamp"
            "Cycle_Phase_ID": [2, 3, 1],
            "Current_L1": [12.5, 13.2, 11.8],
            "Current_L2": [12.3, 13.1, 11.9],
            "Current_L3": [12.4, 13.0, 12.0],
            "Voltage_L_L": [230.5, 229.8, 231.2],
            "Water_Temp_C": [45.0, 50.5, 42.0],
            "Motor_RPM": [1200.0, 1150.0, 1300.0],
            "Water_Flow_L_min": [15.5, 16.2, 14.8],
            "Vibration_mm_s": [2.3, 2.1, 2.5],
            "Water_Pressure_Bar": [3.2, 3.1, 3.3],
            "Vibration_RollingMax_10min": [3.5, 3.2, 3.8]
        })
        
        # FIXED: Changed push_source_name from "machine_features" to "washing_stream_source"
        store.push(
            push_source_name="washing_stream_source",  # This matches the PushSource name in data_sources.py
            df=streaming_data,
            to="online"  # Options: "online", "offline", "online_and_offline"
        )
        
        print("✓ Successfully pushed streaming features")
        print(f"Pushed {len(streaming_data)} records to online store")
        
    except Exception as e:
        print(f"✗ Error: {e}")

    # ============================================================================
    # TEST 4: Materialize Batch Features to Online Store
    # ============================================================================
    print("\n[TEST 4] Materializing batch features to online store...")

    try:
        # Materialize features from the last 7 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        store.materialize(
            start_date=start_date,
            end_date=end_date,
            feature_views=["machine_features"]
        )
        
        print("✓ Successfully materialized batch features")
        print(f"Materialized data from {start_date.date()} to {end_date.date()}")
        
    except Exception as e:
        print(f"✗ Error: {e}")

    # ============================================================================
    # TEST 5: Get Historical Features (Offline Store)
    # ============================================================================
    print("\n[TEST 5] Getting historical features from offline store...")

    try:
        # Create entity dataframe with timestamps
        entity_df = pd.DataFrame({
            "Machine_ID": [1, 2, 3],
            "event_timestamp": [
                datetime.now() - timedelta(hours=1),
                datetime.now() - timedelta(hours=2),
                datetime.now() - timedelta(hours=3)
            ]
        })
        
        training_df = store.get_historical_features(
            entity_df=entity_df,
            features=[
                "machine_features:Current_L1",
                "machine_features:Vibration_mm_s",
                "machine_features:Motor_RPM"
            ]
        ).to_df()
        
        print("✓ Successfully retrieved historical features")
        print(f"Retrieved {len(training_df)} rows")
        print("\nSample data:")
        print(training_df.head())
        
    except Exception as e:
        print(f"✗ Error: {e}")

    # ============================================================================
    # TEST 6: List Feature Views
    # ============================================================================
    print("\n[TEST 6] Listing all registered feature views...")

    try:
        feature_views = store.list_feature_views()
        print("✓ Registered Feature Views:")
        for fv in feature_views:
            print(f"  - {fv.name}")
            # FIXED: Handle both Entity objects and string names
            entity_names = [e.name if hasattr(e, 'name') else str(e) for e in fv.entities]
            print(f"    Entities: {entity_names}")
            print(f"    TTL: {fv.ttl}")
            print(f"    Features: {len(fv.schema)} fields")
            print()
            
    except Exception as e:
        print(f"✗ Error: {e}")

    print("=" * 80)
    print("TESTS COMPLETED")
    print("=" * 80)