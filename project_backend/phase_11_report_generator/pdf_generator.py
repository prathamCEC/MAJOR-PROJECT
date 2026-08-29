"""
Professional PDF Report Generator using ReportLab Platypus.

Constructs multi-page clinical-style assessment documents with tabular scorecards,
risk level callouts, scaled Grad-CAM/SHAP visualizations, and safety disclaimers.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as PlatypusImage,
    KeepTogether,
    HRFlowable,
    PageBreak,
)
from reportlab.pdfgen import canvas

from .config import ReportConfig, get_default_report_config
from .report_data import ClinicalReportData, GradCAMItem


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and draw total page numbers and running footers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))

        # Running Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 11 * 72 - 36, "AI Multimodal Retinal Analysis — Clinical-Style Assessment Report")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(54, 11 * 72 - 42, 8.5 * 72 - 54, 11 * 72 - 42)

        # Running Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * 72 - 54, 36, footer_text)
        self.drawString(54, 36, "CONFIDENTIAL & PROPRIETARY — FOR RESEARCH DECISION SUPPORT ONLY")
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(54, 46, 8.5 * 72 - 54, 46)

        self.restoreState()


class ClinicalPDFReportGenerator:
    """
    High-level engine to render ClinicalReportData into a formatted PDF document.
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or get_default_report_config()
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self) -> None:
        """Define specialized typography and paragraph styles."""
        self.title_style = ParagraphStyle(
            "DocTitle",
            parent=self.styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor(self.config.primary_color),
            spaceAfter=4,
        )
        self.subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=self.styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor(self.config.secondary_color),
            spaceAfter=12,
        )
        self.section_heading = ParagraphStyle(
            "SectionHeading",
            parent=self.styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=colors.HexColor(self.config.primary_color),
            spaceBefore=10,
            spaceAfter=6,
        )
        self.body_style = ParagraphStyle(
            "ReportBody",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor(self.config.neutral_dark),
            spaceAfter=6,
        )
        self.bold_label = ParagraphStyle(
            "BoldLabel",
            parent=self.body_style,
            fontName="Helvetica-Bold",
        )
        self.disclaimer_style = ParagraphStyle(
            "DisclaimerStyle",
            parent=self.styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#742A2A"),
        )
        self.table_cell = ParagraphStyle(
            "TableCell",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor(self.config.neutral_dark),
        )
        self.table_header = ParagraphStyle(
            "TableHeader",
            parent=self.table_cell,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor(self.config.primary_color),
        )

    def generate_pdf(
        self,
        report_data: ClinicalReportData,
        output_filepath: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Build and write PDF report to disk.

        Args:
            report_data: Validated ClinicalReportData instance.
            output_filepath: Target PDF file path (optional).

        Returns:
            Resolved Path of the saved PDF.
        """
        # 1. Validate Input Data
        errors = report_data.validate()
        if errors:
            raise ValueError(f"Report data validation failed: {'; '.join(errors)}")

        # 2. Determine Output Path
        if output_filepath:
            pdf_path = Path(output_filepath).resolve()
        else:
            pdf_dir = self.config.get_pdf_dir()
            pdf_path = pdf_dir / f"report_{report_data.patient_id}_{report_data.report_id}.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

        # 3. Initialize Document Template
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            leftMargin=54,
            rightMargin=54,
            topMargin=54,
            bottomMargin=54,
        )

        story = []

        # -------------------------------------------------------------
        # Header & Branding
        # -------------------------------------------------------------
        story.append(Paragraph(self.config.system_title, self.title_style))
        story.append(Paragraph(f"{self.config.report_title} — {self.config.document_version}", self.subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor(self.config.primary_color), spaceAfter=10))

        # Metadata Header Table
        avail_mods = [k.upper() for k, v in report_data.modalities_available.items() if v]
        mods_str = ", ".join(avail_mods) if avail_mods else "Clinical Data Only"

        meta_data = [
            [
                Paragraph("<b>Patient / Sample ID:</b>", self.body_style),
                Paragraph(report_data.patient_id, self.body_style),
                Paragraph("<b>Report ID:</b>", self.body_style),
                Paragraph(report_data.report_id, self.body_style),
            ],
            [
                Paragraph("<b>Assessment Date:</b>", self.body_style),
                Paragraph(report_data.generated_at, self.body_style),
                Paragraph("<b>Input Modalities:</b>", self.body_style),
                Paragraph(mods_str, self.body_style),
            ],
        ]
        meta_table = Table(meta_data, colWidths=[110, 140, 110, 140])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(self.config.neutral_light)),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 10))

        # -------------------------------------------------------------
        # Section 1: Patient Clinical Profile
        # -------------------------------------------------------------
        demo = report_data.patient_demographics
        story.append(Paragraph("1. Patient Clinical Profile & Biomarkers", self.section_heading))

        demo_data = [
            [
                Paragraph("Age / Group", self.table_header),
                Paragraph(str(demo.age_group), self.table_cell),
                Paragraph("Gender", self.table_header),
                Paragraph(str(demo.gender), self.table_cell),
            ],
            [
                Paragraph("Education (Years)", self.table_header),
                Paragraph(str(demo.education_years), self.table_cell),
                Paragraph("BMI (kg/m²)", self.table_header),
                Paragraph(str(demo.bmi), self.table_cell),
            ],
            [
                Paragraph("Hypertension (HTN)", self.table_header),
                Paragraph(str(demo.hypertension), self.table_cell),
                Paragraph("Diabetes (DM2)", self.table_header),
                Paragraph(str(demo.diabetes_type2), self.table_cell),
            ],
            [
                Paragraph("Smoking History", self.table_header),
                Paragraph(str(demo.smoking_status), self.table_cell),
                Paragraph("Alcohol History", self.table_header),
                Paragraph(str(demo.alcohol_status), self.table_cell),
            ],
        ]
        demo_table = Table(demo_data, colWidths=[120, 130, 120, 130])
        demo_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor(self.config.table_header_bg)),
            ("BACKGROUND", (2, 0), (2, -1), colors.HexColor(self.config.table_header_bg)),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(demo_table)
        story.append(Spacer(1, 10))

        # -------------------------------------------------------------
        # Section 2: Image Quality Assessment (Phase 3)
        # -------------------------------------------------------------
        story.append(Paragraph("2. Retinal Image Technical Quality (Phase 3)", self.section_heading))
        iq_data = [
            [
                Paragraph("Modality", self.table_header),
                Paragraph("Status", self.table_header),
                Paragraph("Quality Score", self.table_header),
                Paragraph("Technical Decision", self.table_header),
            ]
        ]
        for mod in ("octa", "octb", "fundus"):
            item = report_data.image_quality.get(mod)
            if item and item.available:
                score_str = f"{item.quality_score:.1f} / 100" if item.quality_score is not None else "N/A"
                dec_color = "#2F855A" if item.decision == "ACCEPT" else "#C53030"
                dec_cell = Paragraph(f"<b><font color='{dec_color}'>{item.decision}</font></b>", self.table_cell)
                iq_data.append([
                    Paragraph(f"<b>{mod.upper()}</b>", self.table_cell),
                    Paragraph("Available", self.table_cell),
                    Paragraph(score_str, self.table_cell),
                    dec_cell,
                ])
            else:
                iq_data.append([
                    Paragraph(f"<b>{mod.upper()}</b>", self.table_cell),
                    Paragraph("<font color='#718096'>Not available</font>", self.table_cell),
                    Paragraph("—", self.table_cell),
                    Paragraph("—", self.table_cell),
                ])

        iq_table = Table(iq_data, colWidths=[100, 120, 130, 150])
        iq_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(self.config.table_header_bg)),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(iq_table)
        story.append(Spacer(1, 12))

        # -------------------------------------------------------------
        # Section 3: Dual Multi-Task Disease Predictions (Phase 8 & 9)
        # -------------------------------------------------------------
        story.append(Paragraph("3. Multi-Task Disease Predictions & Uncertainty Analysis", self.section_heading))

        st = report_data.stroke_assessment
        al = report_data.alzheimer_assessment

        def get_risk_color(risk_cat: str) -> str:
            if "HIGH" in risk_cat:
                return self.config.alert_color
            elif "MODERATE" in risk_cat:
                return self.config.accent_color
            return self.config.success_color

        pred_data = [
            [
                Paragraph("Assessment Parameter", self.table_header),
                Paragraph("Stroke Target", self.table_header),
                Paragraph("Alzheimer's Disease Target", self.table_header),
            ],
            [
                Paragraph("Model Prediction Class", self.table_cell),
                Paragraph(f"Class {st.predicted_class} ({'Positive' if st.predicted_class==1 else 'Negative'})", self.table_cell),
                Paragraph(f"Class {al.predicted_class} ({'Positive' if al.predicted_class==1 else 'Negative'})", self.table_cell),
            ],
            [
                Paragraph("Predicted Probability", self.table_cell),
                Paragraph(f"<b>{st.probability:.4f}</b>", self.table_cell),
                Paragraph(f"<b>{al.probability:.4f}</b>", self.table_cell),
            ],
            [
                Paragraph("Research Risk Category", self.table_cell),
                Paragraph(f"<b><font color='{get_risk_color(st.risk_category)}'>{st.risk_category}</font></b>", self.table_cell),
                Paragraph(f"<b><font color='{get_risk_color(al.risk_category)}'>{al.risk_category}</font></b>", self.table_cell),
            ],
            [
                Paragraph("Model Confidence (Phase 9)", self.table_cell),
                Paragraph(f"{st.confidence_percent:.2f}% ({st.confidence_level})", self.table_cell),
                Paragraph(f"{al.confidence_percent:.2f}% ({al.confidence_level})", self.table_cell),
            ],
            [
                Paragraph("Predictive Uncertainty (Variance)", self.table_cell),
                Paragraph(f"σ² = {st.predictive_variance:.4f} ({st.uncertainty_level})", self.table_cell),
                Paragraph(f"σ² = {al.predictive_variance:.4f} ({al.uncertainty_level})", self.table_cell),
            ],
            [
                Paragraph("Predictive Shannon Entropy", self.table_cell),
                Paragraph(f"H(p) = {st.predictive_entropy:.4f} nats", self.table_cell),
                Paragraph(f"H(p) = {al.predictive_entropy:.4f} nats", self.table_cell),
            ],
        ]

        pred_table = Table(pred_data, colWidths=[180, 160, 160])
        pred_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(self.config.table_header_bg)),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(pred_table)
        story.append(Spacer(1, 12))

        # -------------------------------------------------------------
        # Section 4: Model Explainability — Visual Grad-CAM Panels
        # -------------------------------------------------------------
        story.append(KeepTogether([
            Paragraph("4. Retinal Visual Explainability (Swin Grad-CAM)", self.section_heading),
            Paragraph(
                "Grad-CAM heatmaps highlight salient retinal regions contributing to model logits. "
                "Each panel displays <b>[Original Retinal Scan | Grad-CAM Heatmap | Alpha-Blended Overlay]</b>.",
                self.body_style,
            ),
        ]))

        exp = report_data.explainability
        gradcam_rendered = False

        for target_name, g_dict in [("Stroke", exp.stroke_gradcam), ("Alzheimer's Disease", exp.alzheimer_gradcam)]:
            for mod_name, item in g_dict.items():
                if item and item.status == "SUCCESS" and item.panel_path and Path(item.panel_path).exists():
                    img_p = Path(item.panel_path).resolve()
                    story.append(Paragraph(f"<b>{target_name} Attribution — {mod_name.upper()} Modality</b>", self.body_style))
                    # Scale image to fit page width (500 pt width, proportional height 140 pt)
                    story.append(PlatypusImage(str(img_p), width=480, height=135))
                    story.append(Paragraph(f"<i>Caption: {item.caption}</i>", self.disclaimer_style))
                    story.append(Spacer(1, 6))
                    gradcam_rendered = True

        if not gradcam_rendered:
            story.append(Paragraph("<i>Retinal Grad-CAM visualizations not available for this session.</i>", self.body_style))

        story.append(Spacer(1, 10))

        # -------------------------------------------------------------
        # Section 5: Clinical Feature Attributions (SHAP)
        # -------------------------------------------------------------
        story.append(KeepTogether([
            Paragraph("5. Clinical Feature Attributions (Game-Theoretic SHAP)", self.section_heading),
            Paragraph(
                "Shapley values (SHAP) quantify marginal additive feature contributions to model log-odds.",
                self.body_style,
            ),
        ]))

        # Render SHAP Summary Table for Stroke & Alzheimer's
        shap_table_data = [
            [
                Paragraph("Feature Name", self.table_header),
                Paragraph("Patient Value", self.table_header),
                Paragraph("Stroke SHAP (Impact)", self.table_header),
                Paragraph("Alzheimer's SHAP (Impact)", self.table_header),
            ]
        ]

        st_map = {item.feature_name: item for item in exp.stroke_shap_clinical}
        al_map = {item.feature_name: item for item in exp.alzheimer_shap_clinical}
        all_features = list(dict.fromkeys(list(st_map.keys()) + list(al_map.keys())))

        if all_features:
            for feat in all_features[:8]:  # Show top 8 features
                st_item = st_map.get(feat)
                al_item = al_map.get(feat)
                val_str = str(st_item.patient_value if st_item else (al_item.patient_value if al_item else "—"))

                st_str = f"{st_item.shap_value:+.4f} ({st_item.direction})" if st_item else "—"
                al_str = f"{al_item.shap_value:+.4f} ({al_item.direction})" if al_item else "—"

                shap_table_data.append([
                    Paragraph(feat, self.table_cell),
                    Paragraph(val_str, self.table_cell),
                    Paragraph(st_str, self.table_cell),
                    Paragraph(al_str, self.table_cell),
                ])

            shap_table = Table(shap_table_data, colWidths=[120, 100, 140, 140])
            shap_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(self.config.table_header_bg)),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("PADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(shap_table)
            story.append(Spacer(1, 8))

        # Render SHAP Bar Chart Plots if available
        for plot_lbl, p_path in [("Stroke Clinical SHAP Plot", exp.stroke_shap_plot_path), ("Alzheimer's Clinical SHAP Plot", exp.alzheimer_shap_plot_path)]:
            if p_path and Path(p_path).exists():
                story.append(Paragraph(f"<b>{plot_lbl}</b>", self.body_style))
                story.append(PlatypusImage(str(Path(p_path).resolve()), width=460, height=160))
                story.append(Spacer(1, 6))

        # -------------------------------------------------------------
        # Section 6: Unified Multimodal & Clinical Narrative Summary
        # -------------------------------------------------------------
        story.append(Paragraph("6. Unified Multimodal Synthesis & Findings", self.section_heading))
        story.append(Paragraph(report_data.clinical_summary_text, self.body_style))
        story.append(Spacer(1, 8))

        # -------------------------------------------------------------
        # Section 7: Limitations & Mandatory Safety Disclaimer
        # -------------------------------------------------------------
        story.append(KeepTogether([
            Paragraph("7. Research Limitations & Methodological Constraints", self.section_heading),
            Paragraph(report_data.limitations_text.replace("\n", "<br/>"), self.body_style),
            Spacer(1, 8),
            Paragraph("<b>MANDATORY RESEARCH DISCLAIMER:</b>", self.disclaimer_style),
            Paragraph(report_data.disclaimer_text, self.disclaimer_style),
        ]))

        # Build PDF using NumberedCanvas
        doc.build(story, canvasmaker=NumberedCanvas)
        return pdf_path
