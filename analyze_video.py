# analyze_video.py
import requests
import base64
import cv2
import os
import json
from collections import Counter

API_URL = "https://pawnder-emotion-api-981944193835.us-east4.run.app"

def analyze_video_emotions(video_path, max_frames=10):
    """Analyze a video by extracting frames and getting emotions for each"""
    
    print(f"🎬 Analyzing video: {video_path}")
    print("=" * 60)
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return None
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"📹 Video Info:")
    print(f"   Duration: {duration:.1f} seconds")
    print(f"   Total Frames: {total_frames}")
    print(f"   FPS: {fps:.1f}")
    print(f"   Analyzing every {total_frames // max_frames} frames")
    
    # Calculate frame interval
    frame_interval = max(1, total_frames // max_frames)
    
    results = []
    frame_count = 0
    analyzed_count = 0
    
    print(f"\n🔍 Extracting and analyzing frames...")
    
    while cap.isOpened() and analyzed_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every Nth frame
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            print(f"   Frame {frame_count} (t={timestamp:.1f}s)... ", end="")
            
            try:
                # Convert frame to JPEG bytes
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Send to API
                payload = {"image": frame_base64}
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    frame_result = {
                        'frame_number': frame_count,
                        'timestamp': timestamp,
                        'emotion': result.get('emotion'),
                        'confidence': result.get('confidence'),
                        'emotion_score': result.get('emotion_score'),
                        'all_emotions': result.get('all_emotions', {})
                    }
                    results.append(frame_result)
                    
                    print(f"✅ {result.get('emotion')} ({result.get('emotion_score', 0):.2f})")
                    analyzed_count += 1
                    
                else:
                    print(f"❌ API Error: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
        
        frame_count += 1
    
    cap.release()
    
    if not results:
        print("❌ No frames were successfully analyzed")
        return None
    
    # Analyze results
    print(f"\n📊 VIDEO ANALYSIS RESULTS")
    print("=" * 60)
    
    # Count emotions
    emotions = [r['emotion'] for r in results]
    emotion_counts = Counter(emotions)
    
    # Calculate averages
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    
    # Find dominant emotion
    dominant_emotion = emotion_counts.most_common(1)[0]
    
    print(f"🏆 Dominant Emotion: {dominant_emotion[0]} ({dominant_emotion[1]}/{len(results)} frames)")
    print(f"📈 Average Confidence: {avg_confidence:.3f}")
    
    print(f"\n📋 Emotion Distribution:")
    for emotion, count in emotion_counts.most_common():
        percentage = (count / len(results)) * 100
        print(f"   {emotion}: {count} frames ({percentage:.1f}%)")
    
    print(f"\n⏱️  Timeline:")
    for result in results:
        print(f"   {result['timestamp']:5.1f}s: {result['emotion']} ({result['emotion_score']:.3f})")
    
    # Save detailed results
    output_file = f"video_analysis_{os.path.basename(video_path).replace('.', '_')}.json"
    
    summary = {
        'video_info': {
            'filename': os.path.basename(video_path),
            'duration': duration,
            'total_frames': total_frames,
            'fps': fps,
            'frames_analyzed': len(results)
        },
        'summary': {
            'dominant_emotion': dominant_emotion[0],
            'dominant_emotion_count': dominant_emotion[1],
            'average_confidence': avg_confidence,
            'emotion_distribution': dict(emotion_counts)
        },
        'frame_results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    return summary

def main():
    """Main function to test video analysis"""
    
    # Create test directory if it doesn't exist
    os.makedirs("test_videos", exist_ok=True)
    
    # Look for video files
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
    video_files = []
    
    # Check test_videos directory
    if os.path.exists("test_videos"):
        for file in os.listdir("test_videos"):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join("test_videos", file))
    
    # Check current directory
    for file in os.listdir("."):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(file)
    
    if not video_files:
        print("💡 No video files found!")
        print("   Add a dog video to test_videos/ folder")
        print("   Supported formats: .mp4, .mov, .avi, .mkv")
        return
    
    # Analyze first video found
    video_path = video_files[0]
    print(f"🎯 Found video: {video_path}")
    
    # Ask user for number of frames to analyze
    try:
        max_frames = int(input(f"\nHow many frames to analyze? (default 10): ") or "10")
    except:
        max_frames = 10
    
    # Analyze the video
    result = analyze_video_emotions(video_path, max_frames=max_frames)
    
    if result:
        print(f"\n🎉 Video analysis complete!")
        print(f"🐕 Your dog appears to be: {result['summary']['dominant_emotion']}")

if __name__ == "__main__":
    main()