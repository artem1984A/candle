use candle::{DType, Device, Result};
use candle_nn::{VarMap, Init};
use candle_nn::var_map::ConcurrentVarMap;
use std::sync::{Arc, Barrier, atomic::{AtomicUsize, Ordering}};
use std::thread;
use std::time::{Duration, Instant};

#[test]
fn stress_test_concurrent_access() -> Result<()> {
    let device = Device::Cpu;
    let varmap = Arc::new(VarMap::new());
    let concurrent = Arc::new(ConcurrentVarMap::new());
    
    // Initialize both with same data
    for i in 0..1000 {
        let name = format!("stress_var_{}", i);
        varmap.get((10, 10), &name, Init::Const(i as f64), DType::F32, &device)?;
        concurrent.get((10, 10), &name, Init::Const(i as f64), DType::F32, &device)?;
    }
    
    // Measure performance under concurrent load
    let n_threads = 16;
    let n_ops_per_thread = 10000;
    let barrier = Arc::new(Barrier::new(n_threads));
    
    // Test original VarMap
    let start = Instant::now();
    let mut handles = vec![];
    
    for _ in 0..n_threads {
        let varmap: Arc<VarMap> = Arc::clone(&varmap);
        let barrier = Arc::clone(&barrier);
        let device = device.clone();
        
        let handle = thread::spawn(move || {
            barrier.wait();
            for i in 0..n_ops_per_thread {
                let name = format!("stress_var_{}", i % 1000);
                let _ = varmap.get((10, 10), &name, Init::Const(0.), DType::F32, &device);
            }
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    let mutex_duration = start.elapsed();
    
    // Test ConcurrentVarMap
    let start = Instant::now();
    let mut handles = vec![];
    
    for _ in 0..n_threads {
        let concurrent: Arc<ConcurrentVarMap> = Arc::clone(&concurrent);
        let barrier = Arc::clone(&barrier);
        let device = device.clone();
        
        let handle = thread::spawn(move || {
            barrier.wait();
            for i in 0..n_ops_per_thread {
                let name = format!("stress_var_{}", i % 1000);
                let _ = concurrent.get((10, 10), &name, Init::Const(0.), DType::F32, &device);
            }
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    let rwlock_duration = start.elapsed();
    
    println!("Mutex duration: {:?}", mutex_duration);
    println!("RwLock duration: {:?}", rwlock_duration);
    println!("Performance improvement: {:.2}x", mutex_duration.as_secs_f64() / rwlock_duration.as_secs_f64());
    
    // Verify data integrity
    assert_eq!(varmap.all_vars().len(), concurrent.all_vars().len());
    
    Ok(())
}

#[test]
fn test_memory_consistency() -> Result<()> {
    let device = Device::Cpu;
    let varmap = Arc::new(VarMap::new());
    let writes_completed = Arc::new(AtomicUsize::new(0));
    
    // Writer thread
    let varmap_writer = Arc::clone(&varmap);
    let writes_counter = Arc::clone(&writes_completed);
    let device_clone = device.clone();
    
    let writer = thread::spawn(move || {
        for i in 0..100 {
            let name = format!("consistency_test_{}", i);
            varmap_writer.get((5, 5), &name, Init::Const(i as f64), DType::F32, &device_clone).unwrap();
            writes_counter.fetch_add(1, Ordering::SeqCst);
            thread::sleep(Duration::from_micros(100));
        }
    });
    
    // Reader threads
    let mut readers = vec![];
    for _ in 0..4 {
        let varmap_reader = Arc::clone(&varmap);
        let writes_counter = Arc::clone(&writes_completed);
        
        let reader = thread::spawn(move || {
            let mut last_count = 0;
            loop {
                let current_writes = writes_counter.load(Ordering::SeqCst);
                if current_writes >= 100 {
                    break;
                }
                
                let var_count = varmap_reader.all_vars().len();
                assert!(var_count >= last_count, "Variables disappeared!");
                last_count = var_count;
                
                thread::sleep(Duration::from_micros(50));
            }
        });
        readers.push(reader);
    }
    
    writer.join().unwrap();
    for reader in readers {
        reader.join().unwrap();
    }
    
    Ok(())
}