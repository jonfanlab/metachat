import sqlite3
import os
from uuid import uuid4
import json
from .models import MaterialData

class MaterialDatabase:
    def __init__(self, db_path: str = "material_database/materials.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create tables
        c.executescript('''
            CREATE TABLE IF NOT EXISTS materials (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT,
                type TEXT,
                min_wavelength REAL,
                max_wavelength REAL,
                wavelength_unit TEXT,
                data_type TEXT,
                file_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS material_references (
                id TEXT PRIMARY KEY,
                material_id TEXT,
                year INTEGER,
                title TEXT,
                journal TEXT,
                doi TEXT,
                citation_count INTEGER,
                last_citation_update TIMESTAMP,
                FOREIGN KEY (material_id) REFERENCES materials(id)
            );

            CREATE TABLE IF NOT EXISTS authors (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE
            );

            CREATE TABLE IF NOT EXISTS reference_authors (
                reference_id TEXT,
                author_id TEXT,
                author_order INTEGER,
                FOREIGN KEY (reference_id) REFERENCES material_references(id),
                FOREIGN KEY (author_id) REFERENCES authors(id),
                PRIMARY KEY (reference_id, author_id)
            );

            CREATE TABLE IF NOT EXISTS specs (
                id TEXT PRIMARY KEY,
                material_id TEXT,
                thickness REAL,
                substrate TEXT,
                temperature REAL,
                additional_info TEXT,
                FOREIGN KEY (material_id) REFERENCES materials(id)
            );

            CREATE TABLE IF NOT EXISTS measurements (
                id TEXT PRIMARY KEY,
                material_id TEXT,
                wavelength REAL,
                n REAL,
                k REAL,
                FOREIGN KEY (material_id) REFERENCES materials(id)
            );

            CREATE TABLE IF NOT EXISTS material_formulas (
                id TEXT PRIMARY KEY,
                material_id TEXT,
                formula_type TEXT NOT NULL,  -- e.g., 'formula 2', 'sellmeier'
                wavelength_range_min REAL,
                wavelength_range_max REAL,
                coefficients TEXT,  -- Store as JSON array of coefficients
                FOREIGN KEY (material_id) REFERENCES materials(id)
            );
        ''')
        
        conn.commit()
        conn.close()    
    def add_material(self, material: MaterialData):
        """Add material data to database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        try:
            # Generate UUIDs
            material_id = str(uuid4())
            
            # Insert material with UUID
            c.execute('''
            INSERT INTO materials (
                id, name, category, type,
                min_wavelength, max_wavelength, wavelength_unit,
                data_type, file_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                material_id,
                material.name,
                material.category.value,
                material.type.value,
                material.wavelength_range[0],
                material.wavelength_range[1],
                material.wavelength_unit,
                material.data_type,
                material.file_path
            ))
            
            # Insert reference with UUID
            if material.reference:
                reference_id = str(uuid4())
                c.execute('''
                INSERT INTO material_references (
                    id, material_id, year, title, journal, doi, citation_count, last_citation_update
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    reference_id,
                    material_id,
                    material.reference.get('year'),
                    material.reference.get('title'),
                    material.reference.get('journal'),
                    material.reference.get('doi'),
                    material.reference.get('citation_count'),
                    material.reference.get('last_citation_update')
                ))
                
                # Handle authors with UUIDs
                authors = material.reference.get('authors', [])
                for i, author_name in enumerate(authors):
                    author_name = author_name.strip()
                    if not author_name:
                        continue
                        
                    # First try to get existing author ID
                    c.execute('SELECT id FROM authors WHERE name = ?', (author_name,))
                    result = c.fetchone()
                    
                    if result:
                        author_id = result[0]
                    else:
                        # Only generate new UUID if author doesn't exist
                        author_id = str(uuid4())
                        c.execute('INSERT INTO authors (id, name) VALUES (?, ?)', 
                                 (author_id, author_name))
                    
                    # Link author to reference
                    c.execute('''
                    INSERT INTO reference_authors (reference_id, author_id, author_order)
                    VALUES (?, ?, ?)
                    ''', (reference_id, author_id, i))
            
            # Insert specs
            if material.specs:
                specs_id = str(uuid4())
                c.execute('''
                INSERT INTO specs (
                    id, material_id, thickness, substrate, temperature,
                    additional_info
                ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    specs_id,
                    material_id,
                    material.specs.get('thickness'),
                    material.specs.get('substrate'),
                    material.specs.get('temperature'),
                    str(material.specs.get('additional_info', {}))
                ))
            
            # Get the type from the first DATA entry
            data_type = material.data.get('DATA', [{}])[0].get('type', '')
            
            # Add formula data if present
            if 'formula' in data_type:
                formula_id = str(uuid4())
                formula_data = material.data['DATA'][0]
                wavelength_range = formula_data.get('wavelength_range', '').split()
                coefficients = formula_data.get('coefficients', '').split()
                
                # Convert coefficients to floats and store as JSON string
                coefficients_json = json.dumps([float(c) for c in coefficients])
                
                # Insert formula data
                c.execute('''
                INSERT INTO material_formulas (
                    id, material_id, formula_type, 
                    wavelength_range_min, wavelength_range_max,
                    coefficients
                ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    formula_id,
                    material_id,
                    data_type,
                    float(wavelength_range[0]),
                    float(wavelength_range[1]),
                    coefficients_json
                ))
            
            # For tabulated data, store in measurements
            elif 'tabulated' in data_type:
                c.executemany('''
                INSERT INTO measurements (
                    id, material_id, wavelength, n, k
                ) VALUES (?, ?, ?, ?, ?)
                ''', [(str(uuid4()), material_id, w, n, k) for w, n, k in material.measurements])
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def get_refractive_index(self, material_id: str, wavelength: float) -> float:
        """Get refractive index for a material at a specific wavelength"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        try:
            # First check if material has a formula
            c.execute('''
                SELECT formula_type, wavelength_range_min, wavelength_range_max, coefficients
                FROM material_formulas
                WHERE material_id = ?
            ''', (material_id,))
            
            formula_data = c.fetchone()
            
            if formula_data:
                formula_type, wl_min, wl_max, coefficients_json = formula_data
                
                # Check wavelength range
                if not (wl_min <= wavelength <= wl_max):
                    raise ValueError(f"Wavelength {wavelength} outside valid range [{wl_min}, {wl_max}]")
                
                coefficients = json.loads(coefficients_json)
                
                if formula_type == 'formula 2':
                    return self._compute_formula2(wavelength, coefficients)
                else:
                    raise ValueError(f"Unsupported formula type: {formula_type}")
            
            # If no formula, interpolate from measurements
            c.execute('''
                SELECT wavelength, n
                FROM measurements
                WHERE material_id = ?
                  AND wavelength <= ?
                ORDER BY wavelength DESC
                LIMIT 1
            ''', (material_id, wavelength))
            lower = c.fetchone()
            
            c.execute('''
                SELECT wavelength, n
                FROM measurements
                WHERE material_id = ?
                  AND wavelength >= ?
                ORDER BY wavelength ASC
                LIMIT 1
            ''', (material_id, wavelength))
            upper = c.fetchone()
            
            if lower and upper:
                # Linear interpolation
                return self._interpolate(wavelength, lower[0], upper[0], lower[1], upper[1])
            
            raise ValueError(f"No data available for wavelength {wavelength}")
            
        finally:
            conn.close()
    
    def _compute_formula2(self, wavelength: float, coefficients: list) -> float:
        """Compute refractive index using Schott's formula 2"""
        A0, A1, A2, A3, A4, A5 = coefficients
        lambda_squared = wavelength * wavelength
        
        n_squared = (A0 + 
                    A1 * lambda_squared + 
                    A2 / lambda_squared + 
                    A3 / (lambda_squared * lambda_squared) + 
                    A4 / (lambda_squared * lambda_squared * lambda_squared) + 
                    A5 / (lambda_squared * lambda_squared * lambda_squared * lambda_squared))
        
        return float(n_squared ** 0.5)
    
    def _interpolate(self, x: float, x1: float, x2: float, y1: float, y2: float) -> float:
        """Linear interpolation helper"""
        return y1 + (x - x1) * (y2 - y1) / (x2 - x1)
