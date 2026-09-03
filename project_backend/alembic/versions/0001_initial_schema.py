"""0001_initial_schema

Revision ID: 0001_initial_schema
Revises: 
Create Date: 2026-09-03 19:05:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0001_initial_schema'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. users
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('username', sa.String(length=100), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=False),
        sa.Column('role', sa.Enum('USER', 'ADMIN', name='userrole'), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.text('1')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)

    # 2. refresh_tokens
    op.create_table(
        'refresh_tokens',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('token_hash', sa.String(length=255), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('is_revoked', sa.Boolean(), nullable=False, server_default=sa.text('0')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_refresh_tokens_token_hash'), 'refresh_tokens', ['token_hash'], unique=True)
    op.create_index(op.f('ix_refresh_tokens_user_id'), 'refresh_tokens', ['user_id'], unique=False)

    # 3. patients
    op.create_table(
        'patients',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('patient_code', sa.String(length=100), nullable=False),
        sa.Column('owner_user_id', sa.Integer(), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=True),
        sa.Column('age_group', sa.String(length=50), nullable=False, server_default='O_CD'),
        sa.Column('gender', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('education_years', sa.Float(), nullable=False, server_default='16.0'),
        sa.Column('bmi', sa.Float(), nullable=False, server_default='26.5'),
        sa.Column('obese', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('hypertension', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('diabetes_type2', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('smoking_ever', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('smoking_current', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('alcohol_ever', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('alcohol_current', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['owner_user_id'], ['users.id'], ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_patients_patient_code'), 'patients', ['patient_code'], unique=True)
    op.create_index(op.f('ix_patients_owner_user_id'), 'patients', ['owner_user_id'], unique=False)

    # 4. analysis_sessions
    op.create_table(
        'analysis_sessions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('session_uuid', sa.String(length=100), nullable=False),
        sa.Column('patient_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('status', sa.Enum('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED', name='analysisstatus'), nullable=False),
        sa.Column('modalities_requested', sa.String(length=100), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['patient_id'], ['patients.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_analysis_sessions_session_uuid'), 'analysis_sessions', ['session_uuid'], unique=True)
    op.create_index(op.f('ix_analysis_sessions_patient_id'), 'analysis_sessions', ['patient_id'], unique=False)
    op.create_index(op.f('ix_analysis_sessions_user_id'), 'analysis_sessions', ['user_id'], unique=False)
    op.create_index(op.f('ix_analysis_sessions_status'), 'analysis_sessions', ['status'], unique=False)

    # 5. uploaded_images
    op.create_table(
        'uploaded_images',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('session_id', sa.Integer(), nullable=False),
        sa.Column('patient_id', sa.Integer(), nullable=False),
        sa.Column('modality', sa.Enum('octa', 'octb', 'fundus', name='modalityenum'), nullable=False),
        sa.Column('original_filename', sa.String(length=255), nullable=False),
        sa.Column('storage_path', sa.String(length=512), nullable=False),
        sa.Column('mime_type', sa.String(length=100), nullable=False),
        sa.Column('file_size_bytes', sa.Integer(), nullable=False),
        sa.Column('image_width', sa.Integer(), nullable=False),
        sa.Column('image_height', sa.Integer(), nullable=False),
        sa.Column('sha256_hash', sa.String(length=64), nullable=False),
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('quality_decision', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['patient_id'], ['patients.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['session_id'], ['analysis_sessions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_uploaded_images_session_id'), 'uploaded_images', ['session_id'], unique=False)
    op.create_index(op.f('ix_uploaded_images_patient_id'), 'uploaded_images', ['patient_id'], unique=False)

    # 6. predictions
    op.create_table(
        'predictions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('session_id', sa.Integer(), nullable=False),
        sa.Column('patient_id', sa.Integer(), nullable=False),
        sa.Column('stroke_probability', sa.Float(), nullable=False),
        sa.Column('stroke_risk_tier', sa.Enum('LOW', 'MODERATE', 'HIGH', name='risktierenum'), nullable=False),
        sa.Column('stroke_confidence_percent', sa.Float(), nullable=False),
        sa.Column('stroke_variance', sa.Float(), nullable=False),
        sa.Column('stroke_entropy', sa.Float(), nullable=False),
        sa.Column('alzheimer_probability', sa.Float(), nullable=False),
        sa.Column('alzheimer_risk_tier', sa.Enum('LOW', 'MODERATE', 'HIGH', name='risktierenum'), nullable=False),
        sa.Column('alzheimer_confidence_percent', sa.Float(), nullable=False),
        sa.Column('alzheimer_variance', sa.Float(), nullable=False),
        sa.Column('alzheimer_entropy', sa.Float(), nullable=False),
        sa.Column('overall_risk_level', sa.Enum('LOW', 'MODERATE', 'HIGH', name='risktierenum'), nullable=False),
        sa.Column('shap_summary_json', sa.Text(), nullable=True),
        sa.Column('gradcam_paths_json', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['patient_id'], ['patients.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['session_id'], ['analysis_sessions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('session_id')
    )
    op.create_index(op.f('ix_predictions_patient_id'), 'predictions', ['patient_id'], unique=False)

    # 7. reports
    op.create_table(
        'reports',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('report_id', sa.String(length=100), nullable=False),
        sa.Column('session_id', sa.Integer(), nullable=False),
        sa.Column('patient_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('pdf_path', sa.String(length=512), nullable=False),
        sa.Column('json_path', sa.String(length=512), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['patient_id'], ['patients.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['session_id'], ['analysis_sessions.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('session_id')
    )
    op.create_index(op.f('ix_reports_report_id'), 'reports', ['report_id'], unique=True)
    op.create_index(op.f('ix_reports_patient_id'), 'reports', ['patient_id'], unique=False)
    op.create_index(op.f('ix_reports_user_id'), 'reports', ['user_id'], unique=False)

    # 8. audit_logs
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('action', sa.String(length=100), nullable=False),
        sa.Column('ip_address', sa.String(length=100), nullable=True),
        sa.Column('user_agent', sa.String(length=255), nullable=True),
        sa.Column('details_json', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_audit_logs_action'), 'audit_logs', ['action'], unique=False)
    op.create_index(op.f('ix_audit_logs_created_at'), 'audit_logs', ['created_at'], unique=False)
    op.create_index(op.f('ix_audit_logs_user_id'), 'audit_logs', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_audit_logs_user_id'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_created_at'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_action'), table_name='audit_logs')
    op.drop_table('audit_logs')

    op.drop_index(op.f('ix_reports_user_id'), table_name='reports')
    op.drop_index(op.f('ix_reports_patient_id'), table_name='reports')
    op.drop_index(op.f('ix_reports_report_id'), table_name='reports')
    op.drop_table('reports')

    op.drop_index(op.f('ix_predictions_patient_id'), table_name='predictions')
    op.drop_table('predictions')

    op.drop_index(op.f('ix_uploaded_images_patient_id'), table_name='uploaded_images')
    op.drop_index(op.f('ix_uploaded_images_session_id'), table_name='uploaded_images')
    op.drop_table('uploaded_images')

    op.drop_index(op.f('ix_analysis_sessions_status'), table_name='analysis_sessions')
    op.drop_index(op.f('ix_analysis_sessions_user_id'), table_name='analysis_sessions')
    op.drop_index(op.f('ix_analysis_sessions_patient_id'), table_name='analysis_sessions')
    op.drop_index(op.f('ix_analysis_sessions_session_uuid'), table_name='analysis_sessions')
    op.drop_table('analysis_sessions')

    op.drop_index(op.f('ix_patients_owner_user_id'), table_name='patients')
    op.drop_index(op.f('ix_patients_patient_code'), table_name='patients')
    op.drop_table('patients')

    op.drop_index(op.f('ix_refresh_tokens_user_id'), table_name='refresh_tokens')
    op.drop_index(op.f('ix_refresh_tokens_token_hash'), table_name='refresh_tokens')
    op.drop_table('refresh_tokens')

    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')
