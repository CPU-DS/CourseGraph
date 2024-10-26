use pyo3::prelude::*;
use rand::Rng;

fn iou(box1: (f32, f32, f32, f32), box2: (f32, f32, f32, f32)) -> f32 {
    let x1 = box1.0.max(box2.0);
    let y1 = box1.1.max(box2.1);
    let x2 = box1.2.min(box2.2);
    let y2 = box1.3.min(box2.3);

    let inter_area = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);

    let box1_area = (box1.2 - box1.0) * (box1.3 - box1.1);
    let box2_area = (box2.2 - box2.0) * (box2.3 - box2.1);

    let union_area = box1_area + box2_area - inter_area;

    if union_area != 0.0 {
        inter_area / union_area
    } else {
        0.0
    }
}

fn is_contained(box1: (f32, f32, f32, f32), box2: (f32, f32, f32, f32)) -> bool {
    box1.0 <= box2.0 && box1.1 <= box2.1 && box1.2 >= box2.2 && box1.3 >= box2.3
}

#[pyfunction]
pub fn post_process(
    detections: Vec<(String, (f32, f32, f32, f32))>,
    iou_threshold: f32,
) -> PyResult<Vec<(String, (f32, f32, f32, f32))>> {
    let mut detections = detections;
    let mut filtered_detections = Vec::new();

    while !detections.is_empty() {
        let detection = detections.remove(0);
        let mut keep = true;

        // 用于存储待移除的检测框
        let mut to_remove = Vec::new();

        for other_detection in detections.clone() {
            if iou(detection.1, other_detection.1) > iou_threshold {
                // 随机选择是否移除
                if rand::thread_rng().gen_bool(0.5) {
                    to_remove.push(other_detection);
                } else {
                    keep = false;
                    break;
                }
            } else if is_contained(detection.1, other_detection.1) {
                to_remove.push(other_detection);
            } else if is_contained(other_detection.1, detection.1) {
                keep = false;
                break;
            }
        }

        for item in to_remove {
            detections.retain(|x| x != &item); // 使用引用
        }

        if keep {
            filtered_detections.push(detection);
        }
    }

    Ok(filtered_detections)
}
