from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from image_processor import ImageProcessor
import logging
from routes.batch_routes import batch_bp
from utils.monitoring import system_monitor
from utils.profiler import model_profiler
from utils.report_generator import report_generator
import time
import threading
import schedule

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
CORS(app)

# Register blueprints
app.register_blueprint(batch_bp)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize components
processor = ImageProcessor()
start_time = time.time()

# Scheduled reports
scheduled_reports = {}

def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)

scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/batch')
def serve_batch():
    return send_from_directory('static', 'batch.html')

@app.route('/dashboard')
def serve_dashboard():
    return send_from_directory('static', 'dashboard.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image
            results = processor.analyze_image(filepath)
            
            # Clean up the uploaded file
            os.remove(filepath)
            
            return jsonify(results)
        
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            system_monitor.record_error('processing_error')
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/metrics')
def get_metrics():
    """Get current system and model metrics."""
    try:
        logger.debug("Getting latest metrics...")
        metrics = system_monitor.get_latest_metrics()
        logger.debug(f"Latest metrics: {metrics}")
        
        if metrics is None:
            logger.debug("No latest metrics, collecting new ones...")
            metrics = system_monitor.collect_metrics()
            logger.debug(f"Collected metrics: {metrics}")
        
        if metrics is None:
            logger.error("Failed to collect metrics")
            return jsonify({
                'error': 'Unable to collect metrics'
            }), 500

        # Helper function to safely get Prometheus metric value
        def get_metric_value(metric, default=0):
            try:
                if metric is None:
                    return default
                if hasattr(metric, '_value'):
                    val = metric._value.get() if hasattr(metric._value, 'get') else metric._value
                    return float(val or default)
                return default
            except Exception as e:
                logger.error(f"Error getting metric value: {str(e)}")
                return default

        # Helper function to safely get histogram buckets
        def get_histogram_buckets(histogram, default=None):
            try:
                if histogram is None:
                    return default or []
                if hasattr(histogram, '_buckets'):
                    return [float(bucket.sum or 0) for bucket in histogram._buckets]
                return default or []
            except Exception as e:
                logger.error(f"Error getting histogram buckets: {str(e)}")
                return default or []

        response_data = {
            'uptime': round((time.time() - start_time) / 3600, 2),  # Convert to hours
            'cpu_usage': metrics.get('cpu', {}).get('percent', 0),
            'memory_usage': metrics.get('memory', {}).get('percent', 0),
            'gpu_usage': metrics.get('gpu', [{'memory_allocated': 0, 'memory_reserved': 1}])[0]['memory_allocated'] / metrics.get('gpu', [{'memory_allocated': 0, 'memory_reserved': 1}])[0]['memory_reserved'] * 100,
            'processed_images': get_metric_value(system_monitor.processed_images),
            'avg_inference_time': get_metric_value(getattr(system_monitor.inference_time, 'count', None)),
            'throughput': get_metric_value(system_monitor.processed_images) / ((time.time() - start_time) or 1),
            'inference_history': get_histogram_buckets(system_monitor.inference_time),
            'error_count': get_metric_value(system_monitor.errors),
            'batch_sizes': get_histogram_buckets(system_monitor.batch_size),
            'network': metrics.get('network', {})
        }
        
        logger.debug(f"Response data: {response_data}")
        return jsonify(response_data)
    except Exception as e:
        logger.exception(f"Error in get_metrics: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e),
            'uptime': 0,
            'cpu_usage': 0,
            'memory_usage': 0,
            'gpu_usage': 0,
            'processed_images': 0,
            'avg_inference_time': 0,
            'throughput': 0,
            'inference_history': [],
            'error_count': 0,
            'batch_sizes': [],
            'network': {}
        }), 500

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model': 'Vision Transformer',
        'version': '1.0',
        'batch_processing': True,
        'explainability': True,
        'monitoring': True
    })

@app.route('/api/reports/generate')
def generate_report():
    """Generate a performance report."""
    try:
        format = request.args.get('format', 'pdf')
        metrics = system_monitor.get_latest_metrics()
        
        if metrics is None:
            metrics = system_monitor.collect_metrics()
        
        # Add historical data
        metrics['inference_history'] = [float(b.sum) for b in system_monitor.inference_time._buckets]
        metrics['cpu_history'] = [m['cpu']['percent'] for m in system_monitor.metrics_history]
        metrics['memory_history'] = [m['memory']['percent'] for m in system_monitor.metrics_history]
        metrics['gpu_history'] = [m['gpu'][0]['memory_allocated'] / m['gpu'][0]['memory_reserved'] * 100 
                                if 'gpu' in m and m['gpu'] else 0 
                                for m in system_monitor.metrics_history]
        
        filepath = report_generator.generate_report(metrics, format)
        return jsonify({
            'success': True,
            'filepath': filepath
        })
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/reports/schedule', methods=['POST'])
def schedule_report():
    """Schedule periodic report generation."""
    try:
        data = request.json
        frequency = data.get('frequency')
        time = data.get('time')
        
        if not frequency or not time:
            return jsonify({
                'success': False,
                'error': 'Missing frequency or time'
            }), 400
        
        # Cancel existing schedule if any
        job_id = f"{frequency}_{time}"
        if job_id in scheduled_reports:
            schedule.cancel_job(scheduled_reports[job_id])
        
        # Create new schedule
        if frequency == 'daily':
            job = schedule.every().day.at(time).do(
                report_generator.generate_report,
                system_monitor.get_latest_metrics(),
                'pdf'
            )
        elif frequency == 'weekly':
            job = schedule.every().week.at(time).do(
                report_generator.generate_report,
                system_monitor.get_latest_metrics(),
                'pdf'
            )
        elif frequency == 'monthly':
            job = schedule.every().month.at(time).do(
                report_generator.generate_report,
                system_monitor.get_latest_metrics(),
                'pdf'
            )
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid frequency'
            }), 400
        
        scheduled_reports[job_id] = job
        
        return jsonify({
            'success': True,
            'job_id': job_id
        })
    except Exception as e:
        logger.error(f"Error scheduling report: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/reports/list')
def list_reports():
    """List all generated reports."""
    try:
        reports_dir = os.path.join(os.getcwd(), 'reports')
        if not os.path.exists(reports_dir):
            return jsonify([])
        
        reports = []
        for filename in os.listdir(reports_dir):
            filepath = os.path.join(reports_dir, filename)
            reports.append({
                'filename': filename,
                'filepath': filepath,
                'size': os.path.getsize(filepath),
                'created': os.path.getctime(filepath)
            })
        
        return jsonify(sorted(reports, key=lambda x: x['created'], reverse=True))
    except Exception as e:
        logger.error(f"Error listing reports: {str(e)}")
        return jsonify([])

@app.route('/api/reports/schedules')
def list_schedules():
    """List all scheduled reports."""
    try:
        return jsonify([{
            'job_id': job_id,
            'next_run': job.next_run.isoformat() if job.next_run else None
        } for job_id, job in scheduled_reports.items()])
    except Exception as e:
        logger.error(f"Error listing schedules: {str(e)}")
        return jsonify([])

@app.route('/api/metrics/export')
def export_metrics():
    """Export metrics data."""
    try:
        metrics_type = request.args.get('type', 'all')
        timeframe = request.args.get('timeframe', '1h')
        
        metrics = system_monitor.get_latest_metrics()
        if metrics is None:
            metrics = system_monitor.collect_metrics()
        
        if metrics_type == 'memory':
            data = {
                'timestamp': [],
                'memory_usage': [],
                'gpu_memory': []
            }
            for m in system_monitor.metrics_history:
                data['timestamp'].append(m['timestamp'])
                data['memory_usage'].append(m['memory']['percent'])
                if 'gpu' in m and m['gpu']:
                    data['gpu_memory'].append(
                        m['gpu'][0]['memory_allocated'] / m['gpu'][0]['memory_reserved'] * 100
                    )
                else:
                    data['gpu_memory'].append(0)
        else:
            data = metrics
        
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error exporting metrics: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
