from astropy.table import Table

from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Image, PageBreak, Paragraph
from reportlab.platypus import Table as rlTable
from reportlab.lib.units import inch

import glob


# Create a PDF document


def gen_report_pdf(CATALOG, SPEC2D_FOLDER, Z_FIT_FOLDER, FLUX_FIT_FOLDER,
                   pdf_filename):
    cat = Table.read(CATALOG)
    doc = SimpleDocTemplate(pdf_filename)
    # List of image IDs you want to include in the PDF

    # Create a PDF document
    doc = SimpleDocTemplate(pdf_filename)



    modules = ['A', 'B']

    to_draw = []

    imgsize = 7
    for src in range(len(cat['ID'])):
        cand_id = cat['ID'][src]

        doublet_ratio = cat['f_O3_5008'][src] / cat['f_O3_4960'][src]

        text = f'{cand_id},\t 5008/4960 = {doublet_ratio:0.2f}'

        to_draw.append(Paragraph(text))

        cand_number = cand_id[:-2]
        
        img = Image(f'{SPEC2D_FOLDER}/spectrum_O3candidate_{cand_id}.png',
                    width=0.8*imgsize*inch, height=0.8*imgsize*inch, kind='proportional')

        to_draw.append(img)

        # Look for Redshift fits in both modules
        zfit_list = glob.glob(f'{Z_FIT_FOLDER}/redshift_fit_O3doublet_COLA1_{cand_number}_*')
        if len(zfit_list) == 0:
            zfit_list = glob.glob(f'{Z_FIT_FOLDER}/redshift_fit_O3_5008_COLA1_{cand_number}_*')
        if len(zfit_list) == 0:
            raise Exception('No images found.')

        tab = []
        for mod in modules:
            for zfit_path in zfit_list:
                if zfit_path[-5] == mod:
                    img = Image(zfit_path, width=0.45*imgsize*inch, height=0.45*imgsize*inch,
                                kind='proportional')
                    tab.append(img)
        if len(tab) > 0:
            to_draw.append(rlTable([tab]))

        # Look for line fits in both modules
        flux_fit_list = glob.glob(f'{FLUX_FIT_FOLDER}/COLA1_ID_{cand_number}_*')
        flux_fit_list.sort()
        for mod in modules:
            tab = []
            for flux_fit_path in flux_fit_list:
                if flux_fit_path[-5] == mod:
                    img = Image(flux_fit_path, width=0.4*imgsize*inch, height=0.4*imgsize*inch,
                                kind='proportional')
                    tab.append(img)
            if len(tab) > 0:
                to_draw.append(rlTable([tab]))

        to_draw.append(PageBreak())


    # Build the PDF
    doc.build(to_draw)

if __name__ == '__main__':

    # DOUBLET SEARCH
    CATALOG = '../catalogs/COLA1_O3_fitted_flux.fits'
    SPEC2D_FOLDER = '../VISCHECK_COLA1_O3/VISCHECK_selected_O3'
    Z_FIT_FOLDER = '../spectra/SPECTRA_O3_FINAL/REDSHIFT_FIT'
    FLUX_FIT_FOLDER = '../spectra/SPECTRA_O3_FINAL/FLUX_FIT'
    pdf_filename = '../catalogs/O3_candidates_report.pdf'
        
    gen_report_pdf(CATALOG, SPEC2D_FOLDER, Z_FIT_FOLDER, FLUX_FIT_FOLDER, pdf_filename)


    # SINGLET SEARCH
    CATALOG = '../catalogs/COLA1_O3_fitted_flux_singlet.fits'
    SPEC2D_FOLDER = '../SINGLESEARCH/VISCHECK_NB921'
    Z_FIT_FOLDER = '../spectra/SPECTRA_SINGLET/REDSHIFT_FIT'
    FLUX_FIT_FOLDER = '../spectra/SPECTRA_SINGLET/FLUX_FIT'
    pdf_filename = '../catalogs/O3_candidates_report_singlet.pdf'
        
    gen_report_pdf(CATALOG, SPEC2D_FOLDER, Z_FIT_FOLDER, FLUX_FIT_FOLDER, pdf_filename)