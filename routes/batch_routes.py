from flask import Blueprint, request, jsonify
from utils.batch_processor import batch_processor
from utils.explainability import ModelExplainer
from werkzeug.utils import secure_filename
import os
from PIL import Image

batch_bp = Blueprint('batch', __name__)

@batch_bp.route('/api/batch/submit', methods=['POST'])
def submit_batch():
    """Submit a batch job for processing."""
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files[]')
    if not files:
        return jsonify({'error': 'No files selected'}), 400
    
    try:
        # Save uploaded files
        image_paths = []
        upload_dir = os.path.join('uploads', 'batch')
        os.makedirs(upload_dir, exist_ok=True)
        
        for file in files:
            if file.filename:
                try:
                    # Ensure filename is safe
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(upload_dir, filename)
                    
                    # Save the file
                    file.save(filepath)
                    
                    # Verify the file was saved correctly
                    if not os.path.exists(filepath):
                        raise Exception("File was not saved correctly")
                    
                    # Verify it's a valid image
                    img = Image.open(filepath)
                    img.verify()  # Verify it's a valid image
                    img.close()
                    
                    # Convert to RGB if needed
                    img = Image.open(filepath)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                        img.save(filepath)
                    img.close()
                    
                    image_paths.append(filepath)
                except Exception as e:
                    print(f"Error processing uploaded file {filename}: {e}")
                    continue
        
        if not image_paths:
            return jsonify({'error': 'No valid images were uploaded'}), 400
        
        # Submit batch job
        job_id = batch_processor.submit_job(image_paths)
        
        return jsonify({
            'job_id': job_id,
            'message': f'Batch job submitted with {len(image_paths)} images'
        })
    except Exception as e:
        print(f"Error in submit_batch: {e}")
        return jsonify({'error': str(e)}), 500

@batch_bp.route('/api/batch/status/<job_id>', methods=['GET'])
def get_batch_status(job_id):
    """Get status of a batch job."""
    status = batch_processor.get_job_status(job_id)
    return jsonify(status)

@batch_bp.route('/api/batch/cancel/<job_id>', methods=['POST'])
def cancel_batch(job_id):
    """Cancel a batch job."""
    success = batch_processor.cancel_job(job_id)
    if success:
        return jsonify({'message': 'Job cancelled successfully'})
    return jsonify({'error': 'Failed to cancel job'}), 400

@batch_bp.route('/api/batch/active', methods=['GET'])
def get_active_jobs():
    """Get list of active batch jobs."""
    try:
        jobs = []
        for job_id, job in batch_processor.jobs.items():
            try:
                job_data = batch_processor.get_job_status(job_id)
                if hasattr(job, 'results') and job.results is not None:
                    # Convert results to list if it's not already
                    results = job.results if isinstance(job.results, list) else list(job.results.values())
                    job_data['results'] = results
                jobs.append(job_data)
            except Exception as e:
                print(f"Error processing job {job_id}: {e}")
                continue
        return jsonify(jobs)
    except Exception as e:
        print(f"Error in get_active_jobs: {e}")
        return jsonify({'error': str(e)}), 500

@batch_bp.route('/api/explain/<job_id>/<image_index>', methods=['GET'])
def explain_result(job_id, image_index):
    """Get explanation for a specific result."""
    try:
        # Get the job
        job = batch_processor.jobs.get(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        # Check job status
        if job.status != 'completed':
            return jsonify({'error': 'Job not completed'}), 400
        
        # Convert image index
        try:
            image_index = int(image_index)
            if image_index >= len(job.results):
                return jsonify({'error': 'Invalid image index'}), 400
        except ValueError:
            return jsonify({'error': 'Invalid image index'}), 400
        
        # Get the result
        result = job.results[image_index]
        if not result.get('success', False):
            return jsonify({'error': 'Image processing failed'}), 400
        
        # Get the image path and top prediction
        image_path = result['original_path']
        if not result.get('predictions'):
            return jsonify({'error': 'No predictions available'}), 400
        
        top_prediction = result['predictions'][0]
        predicted_class = top_prediction['class_id']
        
        try:
            # Generate explanation
            from utils.explainability import ModelExplainer
            explainer = ModelExplainer(model=batch_processor.model, transform=batch_processor.transform)
            explanations = explainer.explain_prediction(image_path, predicted_class)
            
            return jsonify({
                'explanations': explanations,
                'top_prediction': {
                    'class_name': top_prediction['class_name'],
                    'probability': top_prediction['probability']
                }
            })
        except Exception as e:
            print(f"Error generating explanation: {e}")
            return jsonify({'error': f'Failed to generate explanation: {str(e)}'}), 500
            
    except Exception as e:
        print(f"Error in explain_result: {e}")
        return jsonify({'error': str(e)}), 500
